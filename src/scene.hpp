#pragma once
#include <vector>
#include "camera.hpp"
#include "physics_module.hpp"

// simple Body : pos.xyz + unused, vel.xyz + mass in w
struct Body {
    glm::vec4 pos;
    glm::vec4 vel;
};

class Scene {
public:
    Scene(const Camera& cam);
    // main.cpp calls this with a VkCommandBuffer
    void updateAndDispatch(VkCommandBuffer cmd);

private:
    std::vector<Body> _bodies;
    Camera            _camera;
};
