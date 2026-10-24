#pragma once

#include <cstdint>

namespace welvet {

// Go parallel.TrainMode iota — keep numeric parity for wire payloads.
enum class TrainMode : std::uint8_t {
  Inherit = 0,
  NormalBP = 1,
  StepBP = 2,
  Tween = 3,
  TweenChain = 4,
  StepTween = 5,
  StepTweenChain = 6,
  MeshBP = 7,
  MeshTween = 8,
  MeshTweenChain = 9,
  TweenSplit = 10,
  StepTweenSplit = 11,
  TweenAlt = 12,
  StepTweenAlt = 13,
  TweenSplitHeadProxy = 14,
  TweenSplitLinear = 15,
  TweenSplitFastProxy = 16,
  TweenSplitLinearCache = 17,
  TweenSplitHeadProxyAsync = 18,
  TweenSplitSparse = 19,
  MeshTweenSplit = 20,
  MeshTweenAlt = 21,
  MeshTweenSplitFastProxy = 22,
  MeshTweenSplitSparse = 23,
  StepTweenSplitHeadProxy = 24,
  StepTweenSplitLinear = 25,
  StepTweenSplitFastProxy = 26,
  StepTweenSplitLinearCache = 27,
  StepTweenSplitHeadProxyAsync = 28,
  StepTweenSplitSparse = 29,
  Freeze = 30,
  Shadow = 31,
  Adversarial = 32,
  Memory = 33,
};

inline TrainMode resolve_train_mode(TrainMode m, TrainMode parent) {
  if (m == TrainMode::Inherit) {
    return parent == TrainMode::Inherit ? TrainMode::NormalBP : parent;
  }
  return m;
}

inline bool train_mode_is_frozen(TrainMode m) {
  return m == TrainMode::Freeze || m == TrainMode::Shadow;
}

inline bool train_mode_is_shadow(TrainMode m) { return m == TrainMode::Shadow; }

enum class TrainFamily : std::uint8_t {
  BP = 0,
  Tween = 1,
  TweenChain = 2,
  TweenSplit = 3,
  TweenAlt = 4,
  Freeze = 5,
};

} // namespace welvet
