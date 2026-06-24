defmodule AiroAgent.GPUTest do
  use ExUnit.Case, async: true

  alias AiroAgent.GPU

  describe "parse/1 (nvidia-smi --format=csv,noheader,nounits)" do
    test "parses a discrete GPU row into numbers, tagged :nvml" do
      assert GPU.parse("1234, 32607, 15, 250.5, 575") == %{
               available: true,
               vram_used_mb: 1234.0,
               vram_total_mb: 32607.0,
               util_pct: 15.0,
               power_draw_w: 250.5,
               power_limit_w: 575.0,
               mem_source: :nvml
             }
    end

    test "GB10/unified row: N/A memory fields parse to nil (fallback fires later)" do
      # GB10 reports no FB memory via NVML; only util + power.draw populate.
      snap = GPU.parse("[N/A], [N/A], 0, 11.33, [N/A]")
      assert snap.available
      assert snap.vram_used_mb == nil
      assert snap.vram_total_mb == nil
      assert snap.util_pct == 0.0
      assert snap.power_draw_w == 11.33
      assert snap.mem_source == :nvml
    end
  end

  describe "parse_meminfo/1 (unified-memory fallback source)" do
    test "computes total + used (MemTotal - MemAvailable) in MB" do
      contents = """
      MemTotal:       131334740 kB
      MemFree:          5000000 kB
      MemAvailable:   100000000 kB
      Buffers:           200000 kB
      """

      assert {total_mb, used_mb} = GPU.parse_meminfo(contents)
      assert_in_delta total_mb, 131_334_740 / 1024, 0.01
      assert_in_delta used_mb, (131_334_740 - 100_000_000) / 1024, 0.01
    end

    test "errors when the required fields are absent" do
      assert GPU.parse_meminfo("SomethingElse: 1 kB\n") == :error
    end
  end

  describe "parse_darwin_mem/2 (macOS unified-memory source)" do
    @vmstat """
    Mach Virtual Memory Statistics: (page size of 16384 bytes)
    Pages free:                          100000.
    Pages active:                        200000.
    Pages inactive:                      150000.
    Pages speculative:                    50000.
    Pages wired down:                    120000.
    Pages purgeable:                      10000.
    Pages occupied by compressor:         80000.
    """

    test "total from hw.memsize; used = (active + wired + compressor) pages × page size" do
      # 32 GiB total.
      assert {total_mb, used_mb} = GPU.parse_darwin_mem("34359738368\n", @vmstat)
      assert_in_delta total_mb, 32_768.0, 0.01
      # (200000 + 120000 + 80000) pages × 16384 B = 400000 × 16384.
      assert_in_delta used_mb, 400_000 * 16_384 / 1_048_576, 0.01
    end

    test "errors when sysctl output isn't a byte count" do
      assert GPU.parse_darwin_mem("nonsense", @vmstat) == :error
    end

    test "errors when vm_stat has no page size header" do
      assert GPU.parse_darwin_mem("34359738368\n", "Pages active: 1.\n") == :error
    end
  end
end
