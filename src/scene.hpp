#pragma once
#include <vector>
#include "camera.hpp"
#include "physics_module.hpp"

// Use the Body struct defined in physics_module.hpp

class Scene {
public:
    Scene(const Camera& cam);
    // main.cpp calls this with a VkCommandBuffer
    void updateAndDispatch(VkCommandBuffer cmd);

private:
    std::vector<Body> _bodies;
    Camera            _camera;
};
