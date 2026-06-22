defmodule AiroAgent.Api.Auth do
  @moduledoc """
  Bearer-token auth. The agent can spawn processes on the GPU host, so it is
  privileged: if a token is configured, every request must present it. With no
  token set, the app binds loopback-only (see runtime.exs) — local dev only.
  """

  import Plug.Conn

  def init(opts), do: opts

  def call(conn, _opts) do
    case Application.get_env(:airo_agent, :api_token) do
      nil ->
        conn

      expected ->
        if bearer(conn) == expected do
          conn
        else
          conn
          |> put_resp_content_type("application/json")
          |> send_resp(401, ~s({"error":"unauthorized"}))
          |> halt()
        end
    end
  end

  defp bearer(conn) do
    case get_req_header(conn, "authorization") do
      ["Bearer " <> token] -> token
      _ -> nil
    end
  end
end
