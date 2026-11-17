import Lake
open Lake DSL

package tally where
  -- no mathlib dependency: core Lean 4 only, so `lake build` works offline
  -- after the toolchain is installed.

@[default_target]
lean_lib Tally where
  srcDir := "."