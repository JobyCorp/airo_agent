defmodule AiroAgentTest do
  use ExUnit.Case
  doctest AiroAgent

  test "greets the world" do
    assert AiroAgent.hello() == :world
  end
end
