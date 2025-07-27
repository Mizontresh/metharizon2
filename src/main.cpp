// src/main.cpp
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <iostream>               // std::cerr
#include <stdexcept>              // std::runtime_error
#include <vector>                 // std::vector
#include <optional>               // std::optional
#include <set>                    // std::set
#include <fstream>                // std::ifstream
#include <cstdlib>                // EXIT_SUCCESS, EXIT_FAILURE
#include <cstdint>                // uint32_t
#include <algorithm>              // std::clamp

// -----------------------------------------------------------------------------
// Config
// -----------------------------------------------------------------------------
const uint32_t WIDTH  = 800;
const uint32_t HEIGHT = 600;

// -----------------------------------------------------------------------------
// Globals
// -----------------------------------------------------------------------------
GLFWwindow*            window;
VkInstance             instance;
VkSurfaceKHR           surface;

VkPhysicalDevice       physicalDevice = VK_NULL_HANDLE;
VkDevice               device;
VkQueue                computeQueue;
VkQueue                presentQueue;

VkSwapchainKHR         swapchain;
std::vector<VkImage>   swapchainImages;
VkFormat               swapchainFormat;
VkExtent2D             swapchainExtent;
std::vector<VkImageView> swapchainImageViews;

VkCommandPool          commandPool;

// Compute side
VkImage                storageImage;
VkDeviceMemory         storageMemory;
VkImageView            storageView;
VkDescriptorSetLayout  descriptorSetLayout;
VkDescriptorPool       descriptorPool;
VkDescriptorSet        descriptorSet;
VkPipelineLayout       pipelineLayout;
VkPipeline             computePipeline;

// -----------------------------------------------------------------------------
// Helpers & Macros
// -----------------------------------------------------------------------------
#define VK_CHECK(fn)                                                \
    do {                                                            \
        VkResult _res = (fn);                                       \
        if (_res != VK_SUCCESS)                                     \
            throw std::runtime_error(                               \
                std::string("Vulkan error at ") + #fn               \
            );                                                      \
    } while(0)

static std::vector<char> readFile(const std::string& path) {
    std::ifstream f(path, std::ios::ate | std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("failed to open " + path);
    size_t size = (size_t)f.tellg();
    std::vector<char> buf(size);
    f.seekg(0);
    f.read(buf.data(), size);
    f.close();
    return buf;
}

// -----------------------------------------------------------------------------
// Queue‐family selection struct
// -----------------------------------------------------------------------------
struct QueueFamilyIndices {
    std::optional<uint32_t> computeFamily;
    std::optional<uint32_t> presentFamily;
    bool complete() const {
        return computeFamily.has_value() && presentFamily.has_value();
    }
};

// -----------------------------------------------------------------------------
// Declarations
// -----------------------------------------------------------------------------
void initWindow();
void createInstance();
void pickPhysicalDevice();
QueueFamilyIndices findQueueFamilies(VkPhysicalDevice dev);
void createSurface();
void createLogicalDevice();
void createSwapchain();
void createImageViews();
void createCommandPool();
void createStorageImage();
void createComputePipeline();
VkCommandBuffer beginSingleTimeCommands();
void endSingleTimeCommands(VkCommandBuffer cmd);
void dispatchComputeAndPresent();
void cleanup();

// -----------------------------------------------------------------------------
// main()
// -----------------------------------------------------------------------------
int main() {
    try {
        initWindow();
        createInstance();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapchain();
        createImageViews();
        createCommandPool();
        createStorageImage();
        createComputePipeline();

        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            dispatchComputeAndPresent();
        }

        vkDeviceWaitIdle(device);
        cleanup();
    }
    catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

// -----------------------------------------------------------------------------
// initWindow()
// -----------------------------------------------------------------------------
void initWindow() {
    if (!glfwInit())
        throw std::runtime_error("Failed to init GLFW");
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window = glfwCreateWindow(WIDTH, HEIGHT, "Metharizon", nullptr, nullptr);
    if (!window)
        throw std::runtime_error("Failed to create GLFW window");
}

// -----------------------------------------------------------------------------
// createInstance()
// -----------------------------------------------------------------------------
void createInstance() {
    VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    appInfo.pApplicationName   = "Metharizon";
    appInfo.applicationVersion = VK_MAKE_VERSION(1,0,0);
    appInfo.pEngineName        = "No Engine";
    appInfo.engineVersion      = VK_MAKE_VERSION(1,0,0);
    appInfo.apiVersion         = VK_API_VERSION_1_0;

    uint32_t extCount = 0;
    auto glfwExts = glfwGetRequiredInstanceExtensions(&extCount);

    VkInstanceCreateInfo createInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    createInfo.pApplicationInfo    = &appInfo;
    createInfo.enabledExtensionCount   = extCount;
    createInfo.ppEnabledExtensionNames = glfwExts;

    VK_CHECK(vkCreateInstance(&createInfo, nullptr, &instance));
}

// -----------------------------------------------------------------------------
// pickPhysicalDevice()
// -----------------------------------------------------------------------------
void pickPhysicalDevice() {
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    if (count==0) throw std::runtime_error("No Vulkan GPUs");

    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance, &count, devices.data());

    for (auto dev: devices) {
        auto qf = findQueueFamilies(dev);
        if (qf.complete()) {
            physicalDevice = dev;
            break;
        }
    }
    if (physicalDevice==VK_NULL_HANDLE)
        throw std::runtime_error("No GPU with compute + present");
}

// -----------------------------------------------------------------------------
// findQueueFamilies()
// -----------------------------------------------------------------------------
QueueFamilyIndices findQueueFamilies(VkPhysicalDevice dev) {
    QueueFamilyIndices indices;
    uint32_t count=0;
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &count, nullptr);
    std::vector<VkQueueFamilyProperties> props(count);
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &count, props.data());

    for (uint32_t i=0; i<count; i++) {
        if (props[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
            indices.computeFamily = i;

        VkBool32 presentSupport = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(dev, i, surface, &presentSupport);
        if (presentSupport)
            indices.presentFamily = i;

        if (indices.complete()) break;
    }
    return indices;
}

// -----------------------------------------------------------------------------
// createSurface()
// -----------------------------------------------------------------------------
void createSurface() {
    VK_CHECK(glfwCreateWindowSurface(instance, window, nullptr, &surface));
}

// -----------------------------------------------------------------------------
// createLogicalDevice()
// -----------------------------------------------------------------------------
void createLogicalDevice() {
    auto indices = findQueueFamilies(physicalDevice);

    std::vector<VkDeviceQueueCreateInfo> queueInfos;
    std::set<uint32_t> families = {
        *indices.computeFamily,
        *indices.presentFamily
    };

    float priority = 1.0f;
    for (auto fam: families) {
        VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qi.queueFamilyIndex = fam;
        qi.queueCount       = 1;
        qi.pQueuePriorities = &priority;
        queueInfos.push_back(qi);
    }

    VkPhysicalDeviceFeatures features{};
    VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    di.queueCreateInfoCount    = (uint32_t)queueInfos.size();
    di.pQueueCreateInfos       = queueInfos.data();
    di.enabledExtensionCount   = 1;
    const char* ext = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
    di.ppEnabledExtensionNames = &ext;
    di.pEnabledFeatures        = &features;

    VK_CHECK(vkCreateDevice(physicalDevice, &di, nullptr, &device));

    vkGetDeviceQueue(device, *indices.computeFamily, 0, &computeQueue);
    vkGetDeviceQueue(device, *indices.presentFamily, 0, &presentQueue);
}

// -----------------------------------------------------------------------------
// createSwapchain()
// -----------------------------------------------------------------------------
void createSwapchain() {
    VkSurfaceCapabilitiesKHR caps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &caps);

    swapchainExtent = (caps.currentExtent.width != UINT32_MAX)
        ? caps.currentExtent
        : VkExtent2D{WIDTH, HEIGHT};

    uint32_t fmtCount=0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &fmtCount, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(fmtCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &fmtCount, formats.data());
    swapchainFormat = formats[0].format;

    VkSwapchainCreateInfoKHR sci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    sci.surface            = surface;
    sci.minImageCount      = caps.minImageCount;
    sci.imageFormat        = swapchainFormat;
    sci.imageColorSpace    = formats[0].colorSpace;
    sci.imageExtent        = swapchainExtent;
    sci.imageArrayLayers   = 1;
    sci.imageUsage         = VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    auto qf = findQueueFamilies(physicalDevice);
    uint32_t familiesArr[] = {*qf.computeFamily, *qf.presentFamily};
    if (*qf.computeFamily != *qf.presentFamily) {
        sci.imageSharingMode      = VK_SHARING_MODE_CONCURRENT;
        sci.queueFamilyIndexCount = 2;
        sci.pQueueFamilyIndices   = familiesArr;
    } else {
        sci.imageSharingMode      = VK_SHARING_MODE_EXCLUSIVE;
    }

    sci.preTransform   = caps.currentTransform;
    sci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    sci.presentMode    = VK_PRESENT_MODE_FIFO_KHR;
    sci.clipped        = VK_TRUE;
    sci.oldSwapchain   = VK_NULL_HANDLE;

    VK_CHECK(vkCreateSwapchainKHR(device, &sci, nullptr, &swapchain));
    vkGetSwapchainImagesKHR(device, swapchain, &fmtCount, nullptr);
    swapchainImages.resize(fmtCount);
    vkGetSwapchainImagesKHR(device, swapchain, &fmtCount, swapchainImages.data());
}

// -----------------------------------------------------------------------------
// createImageViews()
// -----------------------------------------------------------------------------
void createImageViews() {
    swapchainImageViews.resize(swapchainImages.size());
    for (size_t i=0; i<swapchainImages.size(); i++) {
        VkImageViewCreateInfo iv{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        iv.image    = swapchainImages[i];
        iv.viewType = VK_IMAGE_VIEW_TYPE_2D;
        iv.format   = swapchainFormat;
        iv.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        iv.subresourceRange.baseMipLevel   = 0;
        iv.subresourceRange.levelCount     = 1;
        iv.subresourceRange.baseArrayLayer = 0;
        iv.subresourceRange.layerCount     = 1;
        VK_CHECK(vkCreateImageView(device, &iv, nullptr, &swapchainImageViews[i]));
    }
}

// -----------------------------------------------------------------------------
// createCommandPool()
// -----------------------------------------------------------------------------
void createCommandPool() {
    auto qf = findQueueFamilies(physicalDevice);
    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.queueFamilyIndex = *qf.computeFamily;
    cpi.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK(vkCreateCommandPool(device, &cpi, nullptr, &commandPool));
}

// -----------------------------------------------------------------------------
// createStorageImage()
// -----------------------------------------------------------------------------
void createStorageImage() {
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType        = VK_IMAGE_TYPE_2D;
    ici.extent           = {WIDTH, HEIGHT, 1};
    ici.mipLevels        = 1;
    ici.arrayLayers      = 1;
    ici.format           = VK_FORMAT_R8G8B8A8_UNORM;
    ici.tiling           = VK_IMAGE_TILING_OPTIMAL;
    ici.initialLayout    = VK_IMAGE_LAYOUT_UNDEFINED;
    ici.usage            = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    ici.samples          = VK_SAMPLE_COUNT_1_BIT;
    ici.sharingMode      = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateImage(device, &ici, nullptr, &storageImage));

    VkMemoryRequirements memReq{};
    vkGetImageMemoryRequirements(device, storageImage, &memReq);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize  = memReq.size;

    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);
    for (uint32_t i=0; i<memProps.memoryTypeCount; i++) {
        if ((memReq.memoryTypeBits & (1<<i)) &&
            (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
        {
            mai.memoryTypeIndex = i;
            break;
        }
    }
    VK_CHECK(vkAllocateMemory(device, &mai, nullptr, &storageMemory));
    VK_CHECK(vkBindImageMemory(device, storageImage, storageMemory, 0));

    VkImageViewCreateInfo iv{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    iv.image    = storageImage;
    iv.viewType = VK_IMAGE_VIEW_TYPE_2D;
    iv.format   = VK_FORMAT_R8G8B8A8_UNORM;
    iv.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    iv.subresourceRange.baseMipLevel   = 0;
    iv.subresourceRange.levelCount     = 1;
    iv.subresourceRange.baseArrayLayer = 0;
    iv.subresourceRange.layerCount     = 1;
    VK_CHECK(vkCreateImageView(device, &iv, nullptr, &storageView));
}

// -----------------------------------------------------------------------------
// createComputePipeline()
// -----------------------------------------------------------------------------
void createComputePipeline() {
    // Descriptor layout
    VkDescriptorSetLayoutBinding b{};
    b.binding         = 0;
    b.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    b.descriptorCount = 1;
    b.stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo dsli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dsli.bindingCount = 1;
    dsli.pBindings    = &b;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &dsli, nullptr, &descriptorSetLayout));

    // Pipeline layout
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1;
    pli.pSetLayouts    = &descriptorSetLayout;
    VK_CHECK(vkCreatePipelineLayout(device, &pli, nullptr, &pipelineLayout));

    // Descriptor pool & set
    VkDescriptorPoolSize ps{};
    ps.type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    ps.descriptorCount = 1;
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets       = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &ps;
    VK_CHECK(vkCreateDescriptorPool(device, &dpci, nullptr, &descriptorPool));

    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool     = descriptorPool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &descriptorSetLayout;
    VK_CHECK(vkAllocateDescriptorSets(device, &dsai, &descriptorSet));

    VkDescriptorImageInfo dii{};
    dii.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    dii.imageView   = storageView;
    dii.sampler     = VK_NULL_HANDLE;
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet           = descriptorSet;
    w.dstBinding       = 0;
    w.descriptorCount  = 1;
    w.descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    w.pImageInfo       = &dii;
    vkUpdateDescriptorSets(device, 1, &w, 0, nullptr);

    // Shader module
    auto code = readFile("../shaders/comp.spv");
    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = code.size();
    smci.pCode    = reinterpret_cast<const uint32_t*>(code.data());
    VkShaderModule shaderModule;
    VK_CHECK(vkCreateShaderModule(device, &smci, nullptr, &shaderModule));

    // Compute pipeline
    VkPipelineShaderStageCreateInfo pss{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    pss.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    pss.module = shaderModule;
    pss.pName  = "main";

    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage  = pss;
    cpci.layout = pipelineLayout;
    VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, nullptr, &computePipeline));

    vkDestroyShaderModule(device, shaderModule, nullptr);
}

// -----------------------------------------------------------------------------
// One–shot command buffers
// -----------------------------------------------------------------------------
VkCommandBuffer beginSingleTimeCommands() {
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandPool        = commandPool;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(device, &ai, &cmd);

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    return cmd;
}

void endSingleTimeCommands(VkCommandBuffer cmd) {
    vkEndCommandBuffer(cmd);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &cmd;
    vkQueueSubmit(computeQueue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(computeQueue);
    vkFreeCommandBuffers(device, commandPool, 1, &cmd);
}

// -----------------------------------------------------------------------------
// dispatchComputeAndPresent()
// -----------------------------------------------------------------------------
void dispatchComputeAndPresent() {
    // 1) Compute pass → writes to storageImage (GENERAL)
    VkCommandBuffer cmd = beginSingleTimeCommands();

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
    vkCmdBindDescriptorSets(
        cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        pipelineLayout, 0, 1, &descriptorSet, 0, nullptr
    );
    vkCmdDispatch(cmd,
        (WIDTH + 15)/16,
        (HEIGHT + 15)/16,
        1
    );

    // 2) Barrier: GENERAL → TRANSFER_SRC on storageImage
    VkImageMemoryBarrier bar1{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    bar1.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    bar1.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    bar1.oldLayout     = VK_IMAGE_LAYOUT_GENERAL;
    bar1.newLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    bar1.image         = storageImage;
    bar1.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    bar1.subresourceRange.baseMipLevel   = 0;
    bar1.subresourceRange.levelCount     = 1;
    bar1.subresourceRange.baseArrayLayer = 0;
    bar1.subresourceRange.layerCount     = 1;
    vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr,
        0, nullptr,
        1, &bar1
    );

    // 3) Acquire next swapchain image
    uint32_t idx;
    vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, VK_NULL_HANDLE, VK_NULL_HANDLE, &idx);

    // 4) Barrier: PRESENT → TRANSFER_DST on swapImage
    VkImageMemoryBarrier bar2 = bar1;
    bar2.srcAccessMask = 0;
    bar2.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    bar2.oldLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
    bar2.newLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    bar2.image         = swapchainImages[idx];
    vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,0,nullptr,0,nullptr,1,&bar2
    );

    // 5) Copy storageImage → swapImage
    VkImageCopy copyRegion{};
    copyRegion.srcSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.srcSubresource.layerCount     = 1;
    copyRegion.srcOffset                     = {0,0,0};
    copyRegion.dstSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.dstSubresource.layerCount     = 1;
    copyRegion.dstOffset                     = {0,0,0};
    copyRegion.extent                        = {WIDTH,HEIGHT,1};
    vkCmdCopyImage(
        cmd,
        storageImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        swapchainImages[idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1, &copyRegion
    );

    // 6) Barrier: TRANSFER_DST → PRESENT
    VkImageMemoryBarrier bar3 = bar2;
    bar3.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    bar3.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    bar3.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    bar3.newLayout     = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        0,0,nullptr,0,nullptr,1,&bar3
    );

    endSingleTimeCommands(cmd);

    // 7) Present
    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pi.swapchainCount = 1;
    pi.pSwapchains    = &swapchain;
    pi.pImageIndices  = &idx;
    vkQueuePresentKHR(presentQueue, &pi);
}

// -----------------------------------------------------------------------------
// cleanup()
// -----------------------------------------------------------------------------
void cleanup() {
    vkDestroyPipeline(device, computePipeline, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

    vkDestroyImageView(device, storageView, nullptr);
    vkFreeMemory(device, storageMemory, nullptr);
    vkDestroyImage(device, storageImage, nullptr);

    vkDestroyCommandPool(device, commandPool, nullptr);

    for (auto iv : swapchainImageViews)
        vkDestroyImageView(device, iv, nullptr);
    vkDestroySwapchainKHR(device, swapchain, nullptr);

    vkDestroyDevice(device, nullptr);
    vkDestroySurfaceKHR(instance, surface, nullptr);
    vkDestroyInstance(instance, nullptr);

    glfwDestroyWindow(window);
    glfwTerminate();
}
