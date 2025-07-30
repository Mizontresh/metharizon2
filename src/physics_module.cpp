#include "physics_module.hpp"
#include <cstring>
#include <fstream>
#include <vector>
#include <array>

// Definitions of all statics:
VkDevice              PhysicsModule::_device        = VK_NULL_HANDLE;
VkQueue               PhysicsModule::_queue         = VK_NULL_HANDLE;
VkCommandPool         PhysicsModule::_pool          = VK_NULL_HANDLE;
VkDescriptorSetLayout PhysicsModule::_dsLayout     = VK_NULL_HANDLE;
VkDescriptorPool      PhysicsModule::_dsPool       = VK_NULL_HANDLE;
VkExtent2D            PhysicsModule::_extent       = {};
VkPipeline            PhysicsModule::_physPipeline = VK_NULL_HANDLE;
VkPipelineLayout      PhysicsModule::_physLayout   = VK_NULL_HANDLE;
VkDescriptorSet       PhysicsModule::_ds           = VK_NULL_HANDLE;
VkBuffer              PhysicsModule::_bodyBuffer   = VK_NULL_HANDLE;
VkDeviceMemory        PhysicsModule::_bodyMemory   = VK_NULL_HANDLE;
VkBuffer              PhysicsModule::_camBuffer    = VK_NULL_HANDLE;
VkDeviceMemory        PhysicsModule::_camMemory    = VK_NULL_HANDLE;

static std::vector<char> readFile(const std::string &path) {
    std::ifstream f(path, std::ios::ate | std::ios::binary);
    if (!f) throw std::runtime_error("Failed to open " + path);
    size_t sz = (size_t)f.tellg();
    std::vector<char> buf(sz);
    f.seekg(0);
    f.read(buf.data(), sz);
    return buf;
}

void PhysicsModule::init(VkDevice device,
                         VkPhysicalDevice physDevice,
                         VkQueue queue,
                         uint32_t queueFamily,
                         VkExtent2D storageExtent)
{
    _device    = device;
    _queue     = queue;
    _extent    = storageExtent;

    // command pool for short-lived command buffers
    VkCommandPoolCreateInfo cp{};
    cp.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cp.queueFamilyIndex = queueFamily;
    vkCreateCommandPool(_device, &cp, nullptr, &_pool);

    // descriptor set layout: bodies SSBO + camera UBO
    VkDescriptorSetLayoutBinding b0{};
    b0.binding = 0;
    b0.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b0.descriptorCount = 1;
    b0.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutBinding b1{};
    b1.binding = 1;
    b1.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    b1.descriptorCount = 1;
    b1.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    std::array<VkDescriptorSetLayoutBinding,2> binds = { b0, b1 };
    VkDescriptorSetLayoutCreateInfo dsci{};
    dsci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsci.bindingCount = (uint32_t)binds.size();
    dsci.pBindings = binds.data();
    vkCreateDescriptorSetLayout(_device, &dsci, nullptr, &_dsLayout);

    VkDescriptorPoolSize ps0{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 };
    VkDescriptorPoolSize ps1{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 };
    std::array<VkDescriptorPoolSize,2> pss = { ps0, ps1 };
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets = 1;
    dpci.poolSizeCount = (uint32_t)pss.size();
    dpci.pPoolSizes = pss.data();
    vkCreateDescriptorPool(_device, &dpci, nullptr, &_dsPool);

    VkDescriptorSetAllocateInfo dsai{};
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = _dsPool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &_dsLayout;
    vkAllocateDescriptorSets(_device, &dsai, &_ds);

    // create buffers
    VkDeviceSize bodySize = sizeof(Body) * 64; // room for 64 bodies
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = bodySize;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    vkCreateBuffer(_device, &bci, nullptr, &_bodyBuffer);

    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(_device, _bodyBuffer, &mr);
    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = mr.size;
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(physDevice, &mp);
    for(uint32_t i=0;i<mp.memoryTypeCount;i++){
        if((mr.memoryTypeBits & (1<<i)) && (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)){
            mai.memoryTypeIndex = i; break; }
    }
    vkAllocateMemory(_device, &mai, nullptr, &_bodyMemory);
    vkBindBufferMemory(_device, _bodyBuffer, _bodyMemory, 0);

    bci.size = sizeof(Camera);
    bci.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    vkCreateBuffer(_device, &bci, nullptr, &_camBuffer);
    vkGetBufferMemoryRequirements(_device, _camBuffer, &mr);
    mai.allocationSize = mr.size;
    for(uint32_t i=0;i<mp.memoryTypeCount;i++){
        if((mr.memoryTypeBits & (1<<i)) && (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)){
            mai.memoryTypeIndex = i; break; }
    }
    vkAllocateMemory(_device, &mai, nullptr, &_camMemory);
    vkBindBufferMemory(_device, _camBuffer, _camMemory, 0);

    VkDescriptorBufferInfo bi0{ _bodyBuffer, 0, bodySize };
    VkDescriptorBufferInfo bi1{ _camBuffer, 0, sizeof(Camera) };

    VkWriteDescriptorSet w0{}; w0.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w0.dstSet = _ds; w0.dstBinding = 0; w0.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w0.descriptorCount = 1; w0.pBufferInfo = &bi0;
    VkWriteDescriptorSet w1{}; w1.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w1.dstSet = _ds; w1.dstBinding = 1; w1.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; w1.descriptorCount = 1; w1.pBufferInfo = &bi1;
    std::array<VkWriteDescriptorSet,2> writes = { w0, w1 };
    vkUpdateDescriptorSets(_device, (uint32_t)writes.size(), writes.data(), 0, nullptr);

    // pipeline
    auto spv = readFile("../shaders/physics.spv");
    VkShaderModuleCreateInfo smci{}; smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = spv.size(); smci.pCode = reinterpret_cast<const uint32_t*>(spv.data());
    VkShaderModule mod; vkCreateShaderModule(_device, &smci, nullptr, &mod);

    VkPipelineLayoutCreateInfo plci{}; plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO; plci.setLayoutCount = 1; plci.pSetLayouts = &_dsLayout;
    vkCreatePipelineLayout(_device, &plci, nullptr, &_physLayout);

    VkComputePipelineCreateInfo cpci{}; cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cpci.stage.module = mod; cpci.stage.pName = "main";
    cpci.layout = _physLayout;
    vkCreateComputePipelines(_device, VK_NULL_HANDLE, 1, &cpci, nullptr, &_physPipeline);

    vkDestroyShaderModule(_device, mod, nullptr);
}

void PhysicsModule::step(const Camera& cam)
{
    // update camera buffer only; bodies remain on GPU
    void* ptr;
    vkMapMemory(_device, _camMemory, 0, sizeof(Camera), 0, &ptr);
    std::memcpy(ptr, &cam, sizeof(Camera));
    vkUnmapMemory(_device, _camMemory);
}

void PhysicsModule::uploadBodies(const std::vector<Body>& bodies)
{
    void* ptr;
    vkMapMemory(_device, _bodyMemory, 0,
                bodies.size()*sizeof(Body), 0, &ptr);
    std::memcpy(ptr, bodies.data(), bodies.size()*sizeof(Body));
    vkUnmapMemory(_device, _bodyMemory);
}

void PhysicsModule::recordDispatch(VkCommandBuffer cmd)
{
    // bind & dispatch the physics pipeline into the given cmd buffer:
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, _physPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            _physLayout, 0, 1, &_ds, 0, nullptr);
    vkCmdDispatch(cmd,
        (uint32_t(_extent.width)+63)/64, 1, 1);
}

void PhysicsModule::cleanup()
{
    // teardown in reverse…
    vkDestroyPipeline(_device, _physPipeline, nullptr);
    vkDestroyPipelineLayout(_device, _physLayout, nullptr);
    vkDestroyBuffer(_device, _bodyBuffer, nullptr);
    vkFreeMemory(_device, _bodyMemory, nullptr);
    vkDestroyBuffer(_device, _camBuffer, nullptr);
    vkFreeMemory(_device, _camMemory, nullptr);
    vkDestroyDescriptorPool(_device, _dsPool, nullptr);
    vkDestroyCommandPool(_device, _pool, nullptr);
}
