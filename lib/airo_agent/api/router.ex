defmodule AiroAgent.Api.Router do
  @moduledoc """
  Control-plane HTTP contract Airo consumes (Model 2). Verbs only — NO inference
  proxy. Airo reads `/inventory` for the shelf, calls `/load`/`/unload` to place
  models into **slots**, and routes inference *directly* to a slot's `base_url`.

      GET  /health
      GET  /inventory          -> { models: [ModelRef] }
      GET  /slots              -> { slots: [SlotInfo] }
      GET  /gpu                -> telemetry snapshot
      POST /inventory/refresh  -> rescan local catalog
      POST /load   {model, slot, profile?}  -> SlotInfo
      POST /unload {slot}                   -> { ok: true }
      GET  /openapi            -> the OpenAPI spec for all of the above

  The published contract Airo codes against is `AiroAgent.Api.Spec`, served as
  JSON at `GET /openapi`.
  """

  use Plug.Router
  alias AiroAgent.{Inventory, Fleet, GPU}

  plug(Plug.Parsers, parsers: [:json], pass: ["application/json"], json_decoder: Jason)
  plug(AiroAgent.Api.Auth)
  plug(OpenApiSpex.Plug.PutApiSpec, module: AiroAgent.Api.Spec)
  plug(:match)
  plug(:dispatch)

  # The machine-readable contract (AiroAgent.Api.Spec), rendered from what
  # PutApiSpec stashed on the conn.
  get("/openapi", do: OpenApiSpex.Plug.RenderSpec.call(conn, []))

  # Accepted launch-profile keys (shared + per-engine). Anything else in a POST
  # /load profile is dropped by atomize_profile/1. `ctx`/`parallel`/`extra_argv`
  # are shared; the rest are engine-specific knobs the adapter understands.
  @profile_keys ~w(
    ctx parallel extra_argv disable_thinking
    ngl n_cpu_moe flash_attn cache_type_k cache_type_v spec_type jinja
    chat_template reasoning_budget reasoning_format ld_library_path
    tensor_parallel_size gpu_memory_utilization dtype quantization
    kv_cache_dtype trust_remote_code
  )

  get("/health", do: json(conn, 200, %{status: "ok"}))

  get "/inventory" do
    json(conn, 200, %{models: Enum.map(Inventory.list(), &model_json/1)})
  end

  post "/inventory/refresh" do
    json(conn, 200, %{models: Enum.map(Inventory.refresh(), &model_json/1)})
  end

  get "/slots" do
    json(conn, 200, %{slots: Enum.map(Fleet.slots(), &Map.from_struct/1)})
  end

  get("/gpu", do: json(conn, 200, GPU.snapshot()))

  # Place a model into a slot (port). Airo owns placement (which slot, evict
  # what); the agent loads/swaps and returns the slot's serving endpoint.
  post "/load" do
    case conn.body_params do
      %{"model" => id, "slot" => slot} when is_integer(slot) ->
        profile = conn.body_params |> Map.get("profile", %{}) |> atomize_profile()

        with {:ok, model} <- Inventory.get(id),
             {:ok, info} <- Fleet.load(model, slot, profile) do
          json(conn, 200, Map.from_struct(info))
        else
          {:error, :not_found} -> json(conn, 404, %{error: "unknown model", model: id})
          {:error, reason} -> json(conn, 422, %{error: inspect(reason)})
        end

      %{"model" => _} ->
        json(conn, 400, %{error: "missing or non-integer :slot"})

      _ ->
        json(conn, 400, %{error: "missing :model"})
    end
  end

  post "/unload" do
    case conn.body_params do
      %{"slot" => slot} when is_integer(slot) ->
        case Fleet.unload(slot) do
          :ok -> json(conn, 200, %{ok: true})
          {:error, reason} -> json(conn, 404, %{error: inspect(reason)})
        end

      _ ->
        json(conn, 400, %{error: "missing or non-integer :slot"})
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
