#include "physics_module.hpp"
#include <cstring>

// Definitions of all statics:
VkDevice              PhysicsModule::_device        = VK_NULL_HANDLE;
VkQueue               PhysicsModule::_queue         = VK_NULL_HANDLE;
VkCommandPool         PhysicsModule::_pool          = VK_NULL_HANDLE;
VkDescriptorSetLayout PhysicsModule::_dsLayout     = VK_NULL_HANDLE;
VkDescriptorPool      PhysicsModule::_dsPool       = VK_NULL_HANDLE;
VkExtent2D            PhysicsModule::_extent       = {};
VkPipeline            PhysicsModule::_physPipeline = VK_NULL_HANDLE;
VkPipelineLayout      PhysicsModule::_physLayout   = VK_NULL_HANDLE;
VkPipeline            PhysicsModule::_rayPipeline  = VK_NULL_HANDLE;
VkPipelineLayout      PhysicsModule::_rayLayout    = VK_NULL_HANDLE;
VkDescriptorSet       PhysicsModule::_ds           = VK_NULL_HANDLE;
VkBuffer              PhysicsModule::_bodyBuffer   = VK_NULL_HANDLE;
VkDeviceMemory        PhysicsModule::_bodyMemory   = VK_NULL_HANDLE;
VkBuffer              PhysicsModule::_camBuffer    = VK_NULL_HANDLE;
VkDeviceMemory        PhysicsModule::_camMemory    = VK_NULL_HANDLE;

void PhysicsModule::init(VkDevice device,
                         VkPhysicalDevice physDevice,
                         VkQueue queue,
                         uint32_t queueFamily,
                         VkDescriptorSetLayout dsLayout,
                         VkDescriptorPool dsPool,
                         VkExtent2D storageExtent)
{
    _device    = device;
    _queue     = queue;
    _dsLayout  = dsLayout;
    _dsPool    = dsPool;
    _extent    = storageExtent;

    // **you must still build your pipelines & allocate _ds, _pool,
    //  _bodyBuffer/_camBuffer here exactly as you did before—this
    //  just shows where to stick that code.**
}

void PhysicsModule::step(const std::vector<Body>& bodies,
                         const Camera& cam)
{
    // 1) map+memcpy bodies -> _bodyMemory
    void* ptr;
    vkMapMemory(_device, _bodyMemory, 0,
                bodies.size()*sizeof(Body), 0, &ptr);
    std::memcpy(ptr, bodies.data(), bodies.size()*sizeof(Body));
    vkUnmapMemory(_device, _bodyMemory);

    // 2) map+memcpy cam -> _camMemory
    vkMapMemory(_device, _camMemory, 0, sizeof(Camera), 0, &ptr);
    std::memcpy(ptr, &cam, sizeof(Camera));
    vkUnmapMemory(_device, _camMemory);

    // 3) dispatch the physics pipeline now:
    vkCmdDispatch( /* you’ll need a one‑off cmd buffer here… or else
                     stash this logic into recordDispatch below. */ );
}

void PhysicsModule::recordDispatch(VkCommandBuffer cmd)
{
    // bind & dispatch both pipelines into the given cmd buffer:
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _physPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            _physLayout, 0, 1, &_ds, 0, nullptr);
    vkCmdDispatch(cmd,
        (uint32_t(_extent.width)+63)/64, 1, 1);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _rayPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            _rayLayout, 0, 1, &_ds, 0, nullptr);
    vkCmdDispatch(cmd,
        (_extent.width+15)/16, (_extent.height+15)/16, 1);
}

void PhysicsModule::cleanup()
{
    // teardown in reverse…
    vkDestroyPipeline(_device, _rayPipeline,  nullptr);
    vkDestroyPipelineLayout(_device, _rayLayout,  nullptr);
    vkDestroyPipeline(_device, _physPipeline, nullptr);
    vkDestroyPipelineLayout(_device, _physLayout, nullptr);
    vkDestroyBuffer(_device, _bodyBuffer, nullptr);
    vkFreeMemory(_device, _bodyMemory, nullptr);
    vkDestroyBuffer(_device, _camBuffer, nullptr);
    vkFreeMemory(_device, _camMemory, nullptr);
    vkDestroyDescriptorPool(_device, _dsPool, nullptr);
    vkDestroyCommandPool(_device, _pool, nullptr);
}
