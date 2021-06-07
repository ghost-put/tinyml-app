#pragma once

constexpr int kNumCols = 128;
constexpr int kNumRows = 128;
constexpr int kNumChannels = 1;

constexpr int kMaxImageSize = kNumCols * kNumRows * kNumChannels;

constexpr int kCategoryCount = 2;
constexpr int kMaskIndex = 1;
constexpr int kNotAMaskIndex = 0;
extern const char* kCategoryLabels[kCategoryCount];
