#pragma once

#include <cstdint>

namespace welvet {

uint16_t float32_to_float16(float f);
float float16_to_float32(uint16_t h);
uint16_t float32_to_bfloat16(float f);
float bfloat16_to_float32(uint16_t h);

uint8_t float32_to_fp8e4m3(float f);
float fp8e4m3_to_float32(uint8_t b);
uint8_t float32_to_fp8e5m2(float f);
float fp8e5m2_to_float32(uint8_t b);

uint8_t float32_to_fp4(float f);
float fp4_to_float32(uint8_t code);

} // namespace welvet
