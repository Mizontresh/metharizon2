#pragma once
#include <glm/glm.hpp>

// exactly matches the Camera in main.cpp:
struct Camera {
    alignas(16) float pos[3];
    alignas(16) float forward[3];
    alignas(16) float up[3];
    alignas(16) float right[3];
};
