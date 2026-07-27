defmodule AiroAgent.Test.LazyAdapter do
  @moduledoc false
  # A stand-in engine adapter used ONLY to prove `AiroAgent.Engine.exports?/3`
  # detects an optional callback on a module that hasn't been loaded yet — the
  # state every adapter is in during the boot-time orphan sweep.
  #
  # It exists as its own file, referenced by nothing else, precisely so the test
  # can `:code.purge/1` it. Purging a real adapter would yank it out from under
  # any test running concurrently, and it must live under `test/support` (not be
  # defined inline in the test) so a .beam is written to disk for
  # `Code.ensure_loaded?/1` to load back.

  def reap_orphans, do: :ok
end
