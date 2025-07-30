#pragma once
#include <vector>
#include <vulkan/vulkan.h>
#include "camera.hpp"

// forward-declare Body so we don’t need scene.hpp here:
struct Body { glm::vec4 pos, vel; };

class PhysicsModule {
public:
    // must match the globals in main.cpp:
    static void init(VkDevice device,
                     VkPhysicalDevice physDevice,
                     VkQueue queue,
                     uint32_t queueFamily,
                     VkDescriptorSetLayout dsLayout,
                     VkDescriptorPool dsPool,
                     VkExtent2D storageExtent);

    // upload & run both pipelines immediately
    static void step(const std::vector<Body>& bodies,
                     const Camera& cam);

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
    static VkPipeline            _rayPipeline;
    static VkPipelineLayout      _rayLayout;
    static VkDescriptorSet       _ds;

    // SSBOs for bodies + camera:
    static VkBuffer              _bodyBuffer;
    static VkDeviceMemory        _bodyMemory;
    static VkBuffer              _camBuffer;
    static VkDeviceMemory        _camMemory;
};
