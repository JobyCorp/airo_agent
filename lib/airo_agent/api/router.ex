defmodule AiroAgent.Api.Router do
  @moduledoc """
  Control-plane HTTP contract Airo consumes. Verbs only — NO inference proxy.
  Airo reads `/inventory` for the shelf, calls `/load`/`/unload` for lifecycle,
  and routes inference *directly* to the `base_url` returned here.

      GET  /health
      GET  /inventory          -> { models: [ModelRef] }
      GET  /running            -> { instances: [InstanceInfo] }
      GET  /gpu                -> telemetry snapshot
      POST /inventory/refresh  -> rescan local catalog
      POST /load   {model, profile?}  -> InstanceInfo
      POST /unload {model}            -> { ok: true }
  """

  use Plug.Router
  alias AiroAgent.{Inventory, Fleet, GPU}

  plug(Plug.Parsers, parsers: [:json], pass: ["application/json"], json_decoder: Jason)
  plug(AiroAgent.Api.Auth)
  plug(:match)
  plug(:dispatch)

  @profile_keys ~w(ctx ngl n_cpu_moe flash_attn cache_type_k cache_type_v spec_type parallel jinja chat_template reasoning_budget reasoning_format ld_library_path extra_argv)

  get("/health", do: json(conn, 200, %{status: "ok"}))

  get "/inventory" do
    json(conn, 200, %{models: Enum.map(Inventory.list(), &model_json/1)})
  end

  post "/inventory/refresh" do
    json(conn, 200, %{models: Enum.map(Inventory.refresh(), &model_json/1)})
  end

  get "/running" do
    json(conn, 200, %{instances: Enum.map(Fleet.running(), &Map.from_struct/1)})
  end

  get("/gpu", do: json(conn, 200, GPU.snapshot()))

  post "/load" do
    case conn.body_params do
      %{"model" => id} ->
        profile = conn.body_params |> Map.get("profile", %{}) |> atomize_profile()

        # Inventory resolves the id → ModelRef (provenance); Fleet owns lifecycle.
        with {:ok, model} <- Inventory.get(id),
             {:ok, info} <- Fleet.load(model, profile) do
          json(conn, 200, Map.from_struct(info))
        else
          {:error, :not_found} -> json(conn, 404, %{error: "unknown model", model: id})
          {:error, reason} -> json(conn, 422, %{error: inspect(reason)})
        end

      _ ->
        json(conn, 400, %{error: "missing :model"})
    end
  end

  post "/unload" do
    case conn.body_params do
      %{"model" => id} ->
        case Fleet.unload(id) do
          :ok -> json(conn, 200, %{ok: true})
          {:error, reason} -> json(conn, 404, %{error: inspect(reason)})
        end

      _ ->
        json(conn, 400, %{error: "missing :model"})
    end
  end

  match(_, do: json(conn, 404, %{error: "not_found"}))

  defp json(conn, status, body) do
    conn
    |> put_resp_content_type("application/json")
    |> send_resp(status, Jason.encode!(body))
  end

  defp model_json(ref) do
    ref
    |> Map.from_struct()
    |> Map.update!(:capabilities, &Enum.map(&1, fn c -> to_string(c) end))
  end

  defp atomize_profile(map) when is_map(map) do
    for {k, v} <- map, k in @profile_keys, into: %{}, do: {String.to_atom(k), v}
  end

  defp atomize_profile(_), do: %{}
end
