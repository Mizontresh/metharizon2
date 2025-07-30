#pragma once
#include <vector>
#include <glm/glm.hpp>
#include <vulkan/vulkan.h>
#include "camera.hpp"

// Body layout shared with GLSL shaders (std430)
struct Body {
    glm::vec4 pos;    // xyz position, w = mass
    glm::vec4 vel;    // xyz velocity
    glm::vec4 angVel; // xyz angular velocity
    glm::vec4 orient; // quaternion
    glm::vec4 extra;  // extra.x = maxIter
};

class PhysicsModule {
public:
    // must match the globals in main.cpp:
    static void init(VkDevice device,
                     VkPhysicalDevice physDevice,
                     VkQueue queue,
                     uint32_t queueFamily,
                     VkExtent2D storageExtent);

    // update camera data each frame
    static void step(const Camera& cam);

    // upload initial body data
    static void uploadBodies(const std::vector<Body>& bodies);

    // record into an existing VkCommandBuffer (for main’s ray‑march pass)
    static void recordDispatch(VkCommandBuffer cmd);

    static void cleanup();

private:
    // stored during init:
    static VkDevice              _device;
    static VkQueue               _queue;
    static VkCommandPool         _pool;
    static VkDescriptorSetLayout _dsLayout;
    static VkDescriptorPool      _dsPool;
    static VkExtent2D            _extent;

    // pipelines / layouts / descriptor set:
    static VkPipeline            _physPipeline;
    static VkPipelineLayout      _physLayout;
    static VkDescriptorSet       _ds;

    // SSBOs for bodies + camera:
    static VkBuffer              _bodyBuffer;
    static VkDeviceMemory        _bodyMemory;
    static VkBuffer              _camBuffer;
    static VkDeviceMemory        _camMemory;
};
