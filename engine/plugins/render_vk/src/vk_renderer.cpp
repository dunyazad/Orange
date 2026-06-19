#include "vk_renderer.h"

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#include <algorithm>
#include <cstdint>
#include <cstring>

#include "generated/grid.frag.spv.h"
#include "generated/grid.vert.spv.h"
#include "generated/mesh.frag.spv.h"
#include "generated/mesh.vert.spv.h"

namespace orange::vk {

namespace {

VkFormat attribFormat(render::AttributeFormat f) {
    switch (f) {
        case render::AttributeFormat::Float1: return VK_FORMAT_R32_SFLOAT;
        case render::AttributeFormat::Float2: return VK_FORMAT_R32G32_SFLOAT;
        case render::AttributeFormat::Float3: return VK_FORMAT_R32G32B32_SFLOAT;
        case render::AttributeFormat::Float4: return VK_FORMAT_R32G32B32A32_SFLOAT;
    }
    return VK_FORMAT_R32G32B32_SFLOAT;
}

// column-major 4x4 multiply: out = a * b
void mat4Mul(const float* a, const float* b, float* out) {
    for (int c = 0; c < 4; ++c)
        for (int r = 0; r < 4; ++r) {
            float s = 0.0f;
            for (int k = 0; k < 4; ++k) s += a[k * 4 + r] * b[c * 4 + k];
            out[c * 4 + r] = s;
        }
}

// column-major 4x4 inverse (for the grid's screen-ray unprojection).
void mat4Inverse(const float* m, float* out) {
    float inv[16];
    inv[0]  =  m[5]*m[10]*m[15] - m[5]*m[11]*m[14] - m[9]*m[6]*m[15] + m[9]*m[7]*m[14] + m[13]*m[6]*m[11] - m[13]*m[7]*m[10];
    inv[4]  = -m[4]*m[10]*m[15] + m[4]*m[11]*m[14] + m[8]*m[6]*m[15] - m[8]*m[7]*m[14] - m[12]*m[6]*m[11] + m[12]*m[7]*m[10];
    inv[8]  =  m[4]*m[9]*m[15]  - m[4]*m[11]*m[13] - m[8]*m[5]*m[15] + m[8]*m[7]*m[13] + m[12]*m[5]*m[11] - m[12]*m[7]*m[9];
    inv[12] = -m[4]*m[9]*m[14]  + m[4]*m[10]*m[13] + m[8]*m[5]*m[14] - m[8]*m[6]*m[13] - m[12]*m[5]*m[10] + m[12]*m[6]*m[9];
    inv[1]  = -m[1]*m[10]*m[15] + m[1]*m[11]*m[14] + m[9]*m[2]*m[15] - m[9]*m[3]*m[14] - m[13]*m[2]*m[11] + m[13]*m[3]*m[10];
    inv[5]  =  m[0]*m[10]*m[15] - m[0]*m[11]*m[14] - m[8]*m[2]*m[15] + m[8]*m[3]*m[14] + m[12]*m[2]*m[11] - m[12]*m[3]*m[10];
    inv[9]  = -m[0]*m[9]*m[15]  + m[0]*m[11]*m[13] + m[8]*m[1]*m[15] - m[8]*m[3]*m[13] - m[12]*m[1]*m[11] + m[12]*m[3]*m[9];
    inv[13] =  m[0]*m[9]*m[14]  - m[0]*m[10]*m[13] - m[8]*m[1]*m[14] + m[8]*m[2]*m[13] + m[12]*m[1]*m[10] - m[12]*m[2]*m[9];
    inv[2]  =  m[1]*m[6]*m[15]  - m[1]*m[7]*m[14]  - m[5]*m[2]*m[15] + m[5]*m[3]*m[14] + m[13]*m[2]*m[7]  - m[13]*m[3]*m[6];
    inv[6]  = -m[0]*m[6]*m[15]  + m[0]*m[7]*m[14]  + m[4]*m[2]*m[15] - m[4]*m[3]*m[14] - m[12]*m[2]*m[7]  + m[12]*m[3]*m[6];
    inv[10] =  m[0]*m[5]*m[15]  - m[0]*m[7]*m[13]  - m[4]*m[1]*m[15] + m[4]*m[3]*m[13] + m[12]*m[1]*m[7]  - m[12]*m[3]*m[5];
    inv[14] = -m[0]*m[5]*m[14]  + m[0]*m[6]*m[13]  + m[4]*m[1]*m[14] - m[4]*m[2]*m[13] - m[12]*m[1]*m[6]  + m[12]*m[2]*m[5];
    inv[3]  = -m[1]*m[6]*m[11]  + m[1]*m[7]*m[10]  + m[5]*m[2]*m[11] - m[5]*m[3]*m[10] - m[9]*m[2]*m[7]   + m[9]*m[3]*m[6];
    inv[7]  =  m[0]*m[6]*m[11]  - m[0]*m[7]*m[10]  - m[4]*m[2]*m[11] + m[4]*m[3]*m[10] + m[8]*m[2]*m[7]   - m[8]*m[3]*m[6];
    inv[11] = -m[0]*m[5]*m[11]  + m[0]*m[7]*m[9]   + m[4]*m[1]*m[11] - m[4]*m[3]*m[9]  - m[8]*m[1]*m[7]   + m[8]*m[3]*m[5];
    inv[15] =  m[0]*m[5]*m[10]  - m[0]*m[6]*m[9]   - m[4]*m[1]*m[10] + m[4]*m[2]*m[9]  + m[8]*m[1]*m[6]   - m[8]*m[2]*m[5];
    float det = m[0]*inv[0] + m[1]*inv[4] + m[2]*inv[8] + m[3]*inv[12];
    if (det == 0.0f) { for (int i = 0; i < 16; ++i) out[i] = (i % 5 == 0) ? 1.0f : 0.0f; return; }
    det = 1.0f / det;
    for (int i = 0; i < 16; ++i) out[i] = inv[i] * det;
}

#define VK_CHECK(expr)                                                        \
    do {                                                                      \
        VkResult _r = (expr);                                                 \
        if (_r != VK_SUCCESS) {                                               \
            SDL_Log("VkRenderer: %s failed (VkResult=%d)", #expr, (int)_r);   \
            return false;                                                     \
        }                                                                     \
    } while (0)
} // namespace

bool VkRenderer::init(const render::InitInfo& info) {
    window_ = static_cast<SDL_Window*>(info.nativeWindow);
    pendingW_ = info.width;
    pendingH_ = info.height;
    vsync_  = info.vsync;

    if (!createInstance())        return false;
    if (!createSurface())         return false;
    if (!pickPhysicalDevice())    return false;
    if (!createDevice())          return false;
    if (!createSwapchain())       return false;
    if (!createDepthResources())  return false;
    if (!createRenderPass())      return false;
    if (!createFramebuffers())    return false;
    if (!createCommands())        return false;
    if (!createSyncObjects())     return false;
    if (!createTextureResources()) return false;
    if (!createGridPipeline())    return false;

    SDL_Log("VkRenderer: initialized (%ux%u, %u swapchain images)",
            swapExtent_.width, swapExtent_.height, (unsigned)swapImages_.size());
    return true;
}

bool VkRenderer::createInstance() {
    Uint32 extCount = 0;
    const char* const* sdlExt = SDL_Vulkan_GetInstanceExtensions(&extCount);
    if (!sdlExt) {
        SDL_Log("VkRenderer: SDL_Vulkan_GetInstanceExtensions failed: %s",
                SDL_GetError());
        return false;
    }
    std::vector<const char*> extensions(sdlExt, sdlExt + extCount);

    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "Orange";
    app.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo = &app;
    ci.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    ci.ppEnabledExtensionNames = extensions.data();

    VK_CHECK(vkCreateInstance(&ci, nullptr, &instance_));
    return true;
}

bool VkRenderer::createSurface() {
    if (!SDL_Vulkan_CreateSurface(window_, instance_, nullptr, &surface_)) {
        SDL_Log("VkRenderer: SDL_Vulkan_CreateSurface failed: %s", SDL_GetError());
        return false;
    }
    return true;
}

bool VkRenderer::pickPhysicalDevice() {
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance_, &count, nullptr);
    if (count == 0) {
        SDL_Log("VkRenderer: no Vulkan physical devices");
        return false;
    }
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance_, &count, devices.data());

    for (auto dev : devices) {
        uint32_t qCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &qCount, nullptr);
        std::vector<VkQueueFamilyProperties> props(qCount);
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &qCount, props.data());

        for (uint32_t i = 0; i < qCount; ++i) {
            VkBool32 present = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(dev, i, surface_, &present);
            if ((props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && present) {
                physical_    = dev;
                queueFamily_ = i;
                VkPhysicalDeviceProperties dp{};
                vkGetPhysicalDeviceProperties(dev, &dp);
                SDL_Log("VkRenderer: using GPU '%s'", dp.deviceName);
                return true;
            }
        }
    }
    SDL_Log("VkRenderer: no suitable graphics+present queue family");
    return false;
}

bool VkRenderer::createDevice() {
    float priority = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = queueFamily_;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;

    const char* deviceExt[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

    // fillModeNonSolid: VK_POLYGON_MODE_LINE/POINT (wireframe + point draw modes).
    // largePoints: gl_PointSize > 1 in the point-mode pipeline.
    VkPhysicalDeviceFeatures features{};
    features.fillModeNonSolid = VK_TRUE;
    features.largePoints = VK_TRUE;

    VkDeviceCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    ci.queueCreateInfoCount = 1;
    ci.pQueueCreateInfos = &qci;
    ci.enabledExtensionCount = 1;
    ci.ppEnabledExtensionNames = deviceExt;
    ci.pEnabledFeatures = &features;

    VK_CHECK(vkCreateDevice(physical_, &ci, nullptr, &device_));
    vkGetDeviceQueue(device_, queueFamily_, 0, &queue_);
    return true;
}

bool VkRenderer::createSwapchain() {
    VkSurfaceCapabilitiesKHR caps{};
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_, surface_, &caps);

    // Surface format: prefer B8G8R8A8 SRGB, else first available.
    uint32_t fmtCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physical_, surface_, &fmtCount, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(fmtCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(physical_, surface_, &fmtCount, formats.data());
    VkSurfaceFormatKHR chosen = formats[0];
    for (const auto& f : formats) {
        if (f.format == VK_FORMAT_B8G8R8A8_SRGB &&
            f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            chosen = f;
            break;
        }
    }
    swapFormat_ = chosen.format;

    // Extent.
    if (caps.currentExtent.width != UINT32_MAX) {
        swapExtent_ = caps.currentExtent;
    } else {
        swapExtent_.width = std::clamp(pendingW_, caps.minImageExtent.width,
                                       caps.maxImageExtent.width);
        swapExtent_.height = std::clamp(pendingH_, caps.minImageExtent.height,
                                        caps.maxImageExtent.height);
    }
    if (swapExtent_.width == 0 || swapExtent_.height == 0) {
        // Minimized; skip for now.
        swapExtent_.width = std::max(1u, swapExtent_.width);
        swapExtent_.height = std::max(1u, swapExtent_.height);
    }

    uint32_t imageCount = caps.minImageCount + 1;
    if (caps.maxImageCount > 0 && imageCount > caps.maxImageCount)
        imageCount = caps.maxImageCount;

    VkSwapchainCreateInfoKHR ci{};
    ci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    ci.surface = surface_;
    ci.minImageCount = imageCount;
    ci.imageFormat = chosen.format;
    ci.imageColorSpace = chosen.colorSpace;
    ci.imageExtent = swapExtent_;
    ci.imageArrayLayers = 1;
    ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.preTransform = caps.currentTransform;
    ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    // vsync on -> FIFO (always supported). vsync off -> prefer MAILBOX (low
    // latency, no tearing), else IMMEDIATE (tearing), else fall back to FIFO.
    VkPresentModeKHR present = VK_PRESENT_MODE_FIFO_KHR;
    if (!vsync_) {
        uint32_t pmCount = 0;
        vkGetPhysicalDeviceSurfacePresentModesKHR(physical_, surface_, &pmCount, nullptr);
        std::vector<VkPresentModeKHR> modes(pmCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(physical_, surface_, &pmCount,
                                                  modes.data());
        bool mailbox = false, immediate = false;
        for (auto m : modes) {
            if (m == VK_PRESENT_MODE_MAILBOX_KHR)   mailbox = true;
            if (m == VK_PRESENT_MODE_IMMEDIATE_KHR) immediate = true;
        }
        present = mailbox ? VK_PRESENT_MODE_MAILBOX_KHR
                          : (immediate ? VK_PRESENT_MODE_IMMEDIATE_KHR
                                       : VK_PRESENT_MODE_FIFO_KHR);
    }
    ci.presentMode = present;
    ci.clipped = VK_TRUE;
    ci.oldSwapchain = VK_NULL_HANDLE;

    VK_CHECK(vkCreateSwapchainKHR(device_, &ci, nullptr, &swapchain_));

    uint32_t actual = 0;
    vkGetSwapchainImagesKHR(device_, swapchain_, &actual, nullptr);
    swapImages_.resize(actual);
    vkGetSwapchainImagesKHR(device_, swapchain_, &actual, swapImages_.data());

    swapViews_.resize(actual);
    for (uint32_t i = 0; i < actual; ++i) {
        VkImageViewCreateInfo vci{};
        vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image = swapImages_[i];
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = swapFormat_;
        vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vci.subresourceRange.levelCount = 1;
        vci.subresourceRange.layerCount = 1;
        VK_CHECK(vkCreateImageView(device_, &vci, nullptr, &swapViews_[i]));
    }
    return true;
}

bool VkRenderer::createRenderPass() {
    VkAttachmentDescription color{};
    color.format = swapFormat_;
    color.samples = VK_SAMPLE_COUNT_1_BIT;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    color.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference ref{};
    ref.attachment = 0;
    ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentDescription depth{};
    depth.format = depthFormat_;
    depth.samples = VK_SAMPLE_COUNT_1_BIT;
    depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depth.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depth.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depthRef{};
    depthRef.attachment = 1;
    depthRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &ref;
    subpass.pDepthStencilAttachment = &depthRef;

    VkSubpassDependency dep{};
    dep.srcSubpass = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass = 0;
    dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                       VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                       VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    VkAttachmentDescription attachments[] = {color, depth};
    VkRenderPassCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    ci.attachmentCount = 2;
    ci.pAttachments = attachments;
    ci.subpassCount = 1;
    ci.pSubpasses = &subpass;
    ci.dependencyCount = 1;
    ci.pDependencies = &dep;

    VK_CHECK(vkCreateRenderPass(device_, &ci, nullptr, &renderPass_));
    return true;
}

bool VkRenderer::createFramebuffers() {
    framebuffers_.resize(swapViews_.size());
    for (size_t i = 0; i < swapViews_.size(); ++i) {
        VkImageView attachments[] = {swapViews_[i], depthView_};
        VkFramebufferCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        ci.renderPass = renderPass_;
        ci.attachmentCount = 2;
        ci.pAttachments = attachments;
        ci.width = swapExtent_.width;
        ci.height = swapExtent_.height;
        ci.layers = 1;
        VK_CHECK(vkCreateFramebuffer(device_, &ci, nullptr, &framebuffers_[i]));
    }
    return true;
}

bool VkRenderer::createCommands() {
    VkCommandPoolCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = queueFamily_;
    VK_CHECK(vkCreateCommandPool(device_, &pci, nullptr, &commandPool_));

    commandBuffers_.resize(kFramesInFlight);
    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = commandPool_;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = kFramesInFlight;
    VK_CHECK(vkAllocateCommandBuffers(device_, &ai, commandBuffers_.data()));
    return true;
}

bool VkRenderer::createSyncObjects() {
    imageAvailable_.resize(kFramesInFlight);
    renderFinished_.resize(kFramesInFlight);
    inFlight_.resize(kFramesInFlight);

    VkSemaphoreCreateInfo sci{};
    sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    VkFenceCreateInfo fci{};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (int i = 0; i < kFramesInFlight; ++i) {
        VK_CHECK(vkCreateSemaphore(device_, &sci, nullptr, &imageAvailable_[i]));
        VK_CHECK(vkCreateSemaphore(device_, &sci, nullptr, &renderFinished_[i]));
        VK_CHECK(vkCreateFence(device_, &fci, nullptr, &inFlight_[i]));
    }
    return true;
}

bool VkRenderer::createDepthResources() {
    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = depthFormat_;
    ici.extent = {swapExtent_.width, swapExtent_.height, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VK_CHECK(vkCreateImage(device_, &ici, nullptr, &depthImage_));

    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(device_, depthImage_, &req);
    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = findMemoryType(req.memoryTypeBits,
                                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(device_, &ai, nullptr, &depthMemory_));
    vkBindImageMemory(device_, depthImage_, depthMemory_, 0);

    VkImageViewCreateInfo vci{};
    vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image = depthImage_;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = depthFormat_;
    vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    vci.subresourceRange.levelCount = 1;
    vci.subresourceRange.layerCount = 1;
    VK_CHECK(vkCreateImageView(device_, &vci, nullptr, &depthView_));
    return true;
}

void VkRenderer::destroyDepthResources() {
    if (depthView_)   vkDestroyImageView(device_, depthView_, nullptr);
    if (depthImage_)  vkDestroyImage(device_, depthImage_, nullptr);
    if (depthMemory_) vkFreeMemory(device_, depthMemory_, nullptr);
    depthView_   = VK_NULL_HANDLE;
    depthImage_  = VK_NULL_HANDLE;
    depthMemory_ = VK_NULL_HANDLE;
}

bool VkRenderer::createPipeline(const render::VertexLayout& layout) {
    auto makeModule = [&](const uint32_t* code, size_t bytes) {
        VkShaderModuleCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        ci.codeSize = bytes;
        ci.pCode = code;
        VkShaderModule m = VK_NULL_HANDLE;
        vkCreateShaderModule(device_, &ci, nullptr, &m);
        return m;
    };
    VkShaderModule vs = makeModule(g_meshVertSpv, sizeof(g_meshVertSpv));
    VkShaderModule fs = makeModule(g_meshFragSpv, sizeof(g_meshFragSpv));
    if (!vs || !fs) {
        SDL_Log("VkRenderer: failed to create shader modules");
        return false;
    }

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vs;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fs;
    stages[1].pName = "main";

    // Vertex input derived from the mesh's VertexLayout.
    VkVertexInputBindingDescription binding{};
    binding.binding = 0;
    binding.stride = layout.stride;
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attrs[8]{};
    for (uint32_t i = 0; i < layout.attributeCount; ++i) {
        attrs[i].location = layout.attributes[i].location;
        attrs[i].binding = 0;
        attrs[i].format = attribFormat(layout.attributes[i].format);
        attrs[i].offset = layout.attributes[i].offset;
    }

    VkPipelineVertexInputStateCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vi.vertexBindingDescriptionCount = 1;
    vi.pVertexBindingDescriptions = &binding;
    vi.vertexAttributeDescriptionCount = layout.attributeCount;
    vi.pVertexAttributeDescriptions = attrs;

    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vp{};
    vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vp.viewportCount = 1;
    vp.scissorCount = 1;  // both dynamic

    VkPipelineRasterizationStateCreateInfo rs{};
    rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_NONE;  // matches the GL path (depth-sorted, no cull)
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;
    // Solid fill is biased slightly back so wireframe-over-solid edges sit on top
    // (the GL path uses glPolygonOffset for the same effect).
    rs.depthBiasEnable = VK_TRUE;
    rs.depthBiasConstantFactor = 1.0f;
    rs.depthBiasSlopeFactor = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo ds{};
    ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    ds.depthTestEnable = VK_TRUE;
    ds.depthWriteEnable = VK_TRUE;
    ds.depthCompareOp = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState cba{};
    cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                         VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    cba.blendEnable = VK_TRUE;  // alpha blending for textured text
    cba.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    cba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    cba.colorBlendOp = VK_BLEND_OP_ADD;
    cba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    cba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    cba.alphaBlendOp = VK_BLEND_OP_ADD;
    VkPipelineColorBlendStateCreateInfo cb{};
    cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = 1;
    cb.pAttachments = &cba;

    VkDynamicState dynStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyn{};
    dyn.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyn.dynamicStateCount = 2;
    dyn.pDynamicStates = dynStates;

    // Push constants: viewProj (64) + model (64) = 128 bytes (vertex stage), plus
    // a vec4 force-color at offset 128 (fragment stage) for wireframe-over-solid.
    VkPushConstantRange pcr[2]{};
    pcr[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pcr[0].offset = 0;
    pcr[0].size = 128;
    pcr[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    pcr[1].offset = 128;
    pcr[1].size = 16;

    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &descLayout_;
    plci.pushConstantRangeCount = 2;
    plci.pPushConstantRanges = pcr;
    VK_CHECK(vkCreatePipelineLayout(device_, &plci, nullptr, &pipelineLayout_));

    VkGraphicsPipelineCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pci.stageCount = 2;
    pci.pStages = stages;
    pci.pVertexInputState = &vi;
    pci.pInputAssemblyState = &ia;
    pci.pViewportState = &vp;
    pci.pRasterizationState = &rs;
    pci.pMultisampleState = &ms;
    pci.pDepthStencilState = &ds;
    pci.pColorBlendState = &cb;
    pci.pDynamicState = &dyn;
    pci.layout = pipelineLayout_;
    pci.renderPass = renderPass_;
    pci.subpass = 0;

    VkResult r = vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pci,
                                           nullptr, &pipeline_);

    // Sibling pipelines identical except for the polygon mode -- line (wireframe)
    // and point. The host binds the matching one per Helium drawing mode (Tab).
    // These don't need the solid pipeline's depth bias.
    if (r == VK_SUCCESS) {
        rs.depthBiasEnable = VK_FALSE;
        rs.polygonMode = VK_POLYGON_MODE_LINE;
        VkResult rw = vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pci,
                                                nullptr, &pipelineWireframe_);
        if (rw != VK_SUCCESS) {
            SDL_Log("VkRenderer: wireframe pipeline creation failed (%d)", (int)rw);
            pipelineWireframe_ = VK_NULL_HANDLE;  // fall back to solid
        }
        rs.polygonMode = VK_POLYGON_MODE_POINT;
        VkResult rp = vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pci,
                                                nullptr, &pipelinePoint_);
        if (rp != VK_SUCCESS) {
            SDL_Log("VkRenderer: point pipeline creation failed (%d)", (int)rp);
            pipelinePoint_ = VK_NULL_HANDLE;  // fall back to solid
        }
    }

    vkDestroyShaderModule(device_, vs, nullptr);
    vkDestroyShaderModule(device_, fs, nullptr);
    if (r != VK_SUCCESS) {
        SDL_Log("VkRenderer: vkCreateGraphicsPipelines failed (%d)", (int)r);
        return false;
    }
    return true;
}

bool VkRenderer::createGridPipeline() {
    auto makeModule = [&](const uint32_t* code, size_t bytes) {
        VkShaderModuleCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        ci.codeSize = bytes;
        ci.pCode = code;
        VkShaderModule m = VK_NULL_HANDLE;
        vkCreateShaderModule(device_, &ci, nullptr, &m);
        return m;
    };
    VkShaderModule vs = makeModule(g_gridVertSpv, sizeof(g_gridVertSpv));
    VkShaderModule fs = makeModule(g_gridFragSpv, sizeof(g_gridFragSpv));
    if (!vs || !fs) {
        SDL_Log("VkRenderer: failed to create grid shader modules");
        return false;
    }

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vs;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fs;
    stages[1].pName = "main";

    VkPipelineVertexInputStateCreateInfo vi{};  // vertex-less
    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vp{};
    vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vp.viewportCount = 1;
    vp.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rs{};
    rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_NONE;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo ds{};
    ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    ds.depthTestEnable = VK_TRUE;
    ds.depthWriteEnable = VK_TRUE;
    ds.depthCompareOp = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState cba{};
    cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                         VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    cba.blendEnable = VK_TRUE;
    cba.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    cba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    cba.colorBlendOp = VK_BLEND_OP_ADD;
    cba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    cba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    cba.alphaBlendOp = VK_BLEND_OP_ADD;
    VkPipelineColorBlendStateCreateInfo cb{};
    cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = 1;
    cb.pAttachments = &cba;

    VkDynamicState dynStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyn{};
    dyn.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyn.dynamicStateCount = 2;
    dyn.pDynamicStates = dynStates;

    // Push constant: viewProj (0) + invViewProj (64) + upAxis (128), both stages.
    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pcr.offset = 0;
    pcr.size = 132;

    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 0;  // grid samples no texture
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    VK_CHECK(vkCreatePipelineLayout(device_, &plci, nullptr, &gridLayout_));

    VkGraphicsPipelineCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pci.stageCount = 2;
    pci.pStages = stages;
    pci.pVertexInputState = &vi;
    pci.pInputAssemblyState = &ia;
    pci.pViewportState = &vp;
    pci.pRasterizationState = &rs;
    pci.pMultisampleState = &ms;
    pci.pDepthStencilState = &ds;
    pci.pColorBlendState = &cb;
    pci.pDynamicState = &dyn;
    pci.layout = gridLayout_;
    pci.renderPass = renderPass_;
    pci.subpass = 0;

    VkResult r = vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pci,
                                           nullptr, &gridPipeline_);
    vkDestroyShaderModule(device_, vs, nullptr);
    vkDestroyShaderModule(device_, fs, nullptr);
    if (r != VK_SUCCESS) {
        SDL_Log("VkRenderer: grid pipeline creation failed (%d)", (int)r);
        return false;
    }
    return true;
}

void VkRenderer::drawGrid(int upAxis) {
    if (!frameActive_ || gridPipeline_ == VK_NULL_HANDLE) return;
    VkCommandBuffer cmd = commandBuffers_[currentFrame_];
    mat4Inverse(viewProj_, invViewProj_);  // viewProj_ holds the scene's clip*proj*view

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, gridPipeline_);
    VkShaderStageFlags st = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    vkCmdPushConstants(cmd, gridLayout_, st, 0, 64, viewProj_);
    vkCmdPushConstants(cmd, gridLayout_, st, 64, 64, invViewProj_);
    vkCmdPushConstants(cmd, gridLayout_, st, 128, 4, &upAxis);
    vkCmdDraw(cmd, 3, 1, 0, 0);

    // Restore the mesh pipeline so the gizmo/UI overlays (which assume it) work.
    if (pipeline_ != VK_NULL_HANDLE)
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
}

void VkRenderer::destroySwapchain() {
    destroyDepthResources();
    for (auto fb : framebuffers_) vkDestroyFramebuffer(device_, fb, nullptr);
    framebuffers_.clear();
    for (auto v : swapViews_) vkDestroyImageView(device_, v, nullptr);
    swapViews_.clear();
    swapImages_.clear();
    if (swapchain_) {
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        swapchain_ = VK_NULL_HANDLE;
    }
}

bool VkRenderer::recreateSwapchain() {
    vkDeviceWaitIdle(device_);
    destroySwapchain();
    if (!createSwapchain())      return false;
    if (!createDepthResources()) return false;
    if (!createFramebuffers())   return false;
    needsResize_ = false;
    return true;
}

uint32_t VkRenderer::findMemoryType(uint32_t typeBits,
                                    VkMemoryPropertyFlags props) const {
    VkPhysicalDeviceMemoryProperties memProps{};
    vkGetPhysicalDeviceMemoryProperties(physical_, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((typeBits & (1u << i)) &&
            (memProps.memoryTypes[i].propertyFlags & props) == props) {
            return i;
        }
    }
    SDL_Log("VkRenderer: no suitable memory type");
    return 0;
}

render::BufferHandle VkRenderer::createBuffer(const render::BufferDesc& desc) {
    if (desc.size == 0 || device_ == VK_NULL_HANDLE) return render::kInvalidBuffer;

    VkBufferUsageFlags usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    switch (desc.type) {
        case render::BufferType::Vertex:
            usage |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT; break;
        case render::BufferType::Index:
            usage |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT; break;
        case render::BufferType::Uniform:
            usage |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT; break;
    }

    VkBufferObject obj;
    obj.size = desc.size;

    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = desc.size;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(device_, &bi, nullptr, &obj.buffer) != VK_SUCCESS) {
        SDL_Log("VkRenderer: vkCreateBuffer failed");
        return render::kInvalidBuffer;
    }

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(device_, obj.buffer, &req);

    // Host-visible + coherent: simple to upload to without a staging copy.
    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = findMemoryType(req.memoryTypeBits,
                                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (vkAllocateMemory(device_, &ai, nullptr, &obj.memory) != VK_SUCCESS) {
        SDL_Log("VkRenderer: vkAllocateMemory failed");
        vkDestroyBuffer(device_, obj.buffer, nullptr);
        return render::kInvalidBuffer;
    }
    vkBindBufferMemory(device_, obj.buffer, obj.memory, 0);

    if (desc.data) {
        void* mapped = nullptr;
        vkMapMemory(device_, obj.memory, 0, desc.size, 0, &mapped);
        std::memcpy(mapped, desc.data, desc.size);
        vkUnmapMemory(device_, obj.memory);
    }

    render::BufferHandle handle = nextBuffer_++;
    buffers_[handle] = obj;
    return handle;
}

void VkRenderer::updateBuffer(render::BufferHandle handle, const void* data,
                              size_t size, size_t offset) {
    auto it = buffers_.find(handle);
    if (it == buffers_.end() || !data) return;
    const VkBufferObject& obj = it->second;
    if (offset + size > obj.size) {
        SDL_Log("VkRenderer: updateBuffer out of range");
        return;
    }
    void* mapped = nullptr;
    vkMapMemory(device_, obj.memory, offset, size, 0, &mapped);
    std::memcpy(mapped, data, size);
    vkUnmapMemory(device_, obj.memory);
}

void VkRenderer::destroyBuffer(render::BufferHandle handle) {
    auto it = buffers_.find(handle);
    if (it == buffers_.end()) return;
    if (device_) vkDeviceWaitIdle(device_);
    vkDestroyBuffer(device_, it->second.buffer, nullptr);
    vkFreeMemory(device_, it->second.memory, nullptr);
    buffers_.erase(it);
}

bool VkRenderer::createTextureResources() {
    VkSamplerCreateInfo sci{};
    sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sci.magFilter = VK_FILTER_LINEAR;
    sci.minFilter = VK_FILTER_LINEAR;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.maxLod = VK_LOD_CLAMP_NONE;
    VK_CHECK(vkCreateSampler(device_, &sci, nullptr, &sampler_));

    VkDescriptorSetLayoutBinding b{};
    b.binding = 0;
    b.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b.descriptorCount = 1;
    b.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutCreateInfo lci{};
    lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    lci.bindingCount = 1;
    lci.pBindings = &b;
    VK_CHECK(vkCreateDescriptorSetLayout(device_, &lci, nullptr, &descLayout_));

    VkDescriptorPoolSize ps{};
    ps.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    ps.descriptorCount = 64;
    VkDescriptorPoolCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pci.maxSets = 64;
    pci.poolSizeCount = 1;
    pci.pPoolSizes = &ps;
    VK_CHECK(vkCreateDescriptorPool(device_, &pci, nullptr, &descPool_));

    // 1x1 white fallback so untextured draws keep their vertex color.
    const uint8_t white[4] = {255, 255, 255, 255};
    render::TextureDesc wd;
    wd.width = 1; wd.height = 1; wd.pixels = white;
    render::TextureHandle h = createTexture(wd);
    if (h == render::kInvalidTexture) return false;
    whiteSet_ = textures_[h].set;
    return true;
}

render::TextureHandle VkRenderer::createTexture(const render::TextureDesc& desc) {
    if (!desc.pixels || desc.width == 0 || desc.height == 0)
        return render::kInvalidTexture;
    VkDeviceSize sz = static_cast<VkDeviceSize>(desc.width) * desc.height * 4;

    // Staging buffer (host visible) with the pixel data.
    VkBuffer staging = VK_NULL_HANDLE;
    VkDeviceMemory stagingMem = VK_NULL_HANDLE;
    {
        VkBufferCreateInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bi.size = sz;
        bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateBuffer(device_, &bi, nullptr, &staging) != VK_SUCCESS)
            return render::kInvalidTexture;
        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(device_, staging, &req);
        VkMemoryAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = findMemoryType(req.memoryTypeBits,
                                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        vkAllocateMemory(device_, &ai, nullptr, &stagingMem);
        vkBindBufferMemory(device_, staging, stagingMem, 0);
        void* mapped = nullptr;
        vkMapMemory(device_, stagingMem, 0, sz, 0, &mapped);
        std::memcpy(mapped, desc.pixels, static_cast<size_t>(sz));
        vkUnmapMemory(device_, stagingMem);
    }

    VkTexture tex;
    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_R8G8B8A8_UNORM;
    ici.extent = {desc.width, desc.height, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    vkCreateImage(device_, &ici, nullptr, &tex.image);

    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(device_, tex.image, &req);
    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = findMemoryType(req.memoryTypeBits,
                                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vkAllocateMemory(device_, &ai, nullptr, &tex.memory);
    vkBindImageMemory(device_, tex.image, tex.memory, 0);

    // One-time command buffer: transition, copy, transition.
    VkCommandBufferAllocateInfo cai{};
    cai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cai.commandPool = commandPool_;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(device_, &cai, &cmd);
    VkCommandBufferBeginInfo cbi{};
    cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &cbi);

    auto barrier = [&](VkImageLayout oldL, VkImageLayout newL, VkAccessFlags srcA,
                       VkAccessFlags dstA, VkPipelineStageFlags srcS,
                       VkPipelineStageFlags dstS) {
        VkImageMemoryBarrier mb{};
        mb.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        mb.oldLayout = oldL;
        mb.newLayout = newL;
        mb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        mb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        mb.image = tex.image;
        mb.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        mb.srcAccessMask = srcA;
        mb.dstAccessMask = dstA;
        vkCmdPipelineBarrier(cmd, srcS, dstS, 0, 0, nullptr, 0, nullptr, 1, &mb);
    };
    barrier(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT);

    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {desc.width, desc.height, 1};
    vkCmdCopyBufferToImage(cmd, staging, tex.image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    barrier(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

    vkEndCommandBuffer(cmd);
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    vkQueueSubmit(queue_, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue_);
    vkFreeCommandBuffers(device_, commandPool_, 1, &cmd);
    vkDestroyBuffer(device_, staging, nullptr);
    vkFreeMemory(device_, stagingMem, nullptr);

    // View + descriptor set.
    VkImageViewCreateInfo vci{};
    vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image = tex.image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = VK_FORMAT_R8G8B8A8_UNORM;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCreateImageView(device_, &vci, nullptr, &tex.view);

    VkDescriptorSetAllocateInfo dai{};
    dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dai.descriptorPool = descPool_;
    dai.descriptorSetCount = 1;
    dai.pSetLayouts = &descLayout_;
    vkAllocateDescriptorSets(device_, &dai, &tex.set);

    VkDescriptorImageInfo dii{};
    dii.sampler = sampler_;
    dii.imageView = tex.view;
    dii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkWriteDescriptorSet w{};
    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstSet = tex.set;
    w.dstBinding = 0;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w.pImageInfo = &dii;
    vkUpdateDescriptorSets(device_, 1, &w, 0, nullptr);

    render::TextureHandle handle = nextTexture_++;
    textures_[handle] = tex;
    return handle;
}

void VkRenderer::destroyTexture(render::TextureHandle handle) {
    auto it = textures_.find(handle);
    if (it == textures_.end()) return;
    vkDeviceWaitIdle(device_);
    vkFreeDescriptorSets(device_, descPool_, 1, &it->second.set);
    vkDestroyImageView(device_, it->second.view, nullptr);
    vkDestroyImage(device_, it->second.image, nullptr);
    vkFreeMemory(device_, it->second.memory, nullptr);
    textures_.erase(it);
}

render::MeshHandle VkRenderer::createMesh(const render::MeshDesc& desc) {
    if (buffers_.find(desc.vertexBuffer) == buffers_.end() ||
        desc.vertexCount == 0) {
        SDL_Log("VkRenderer: createMesh needs a valid vertex buffer");
        return render::kInvalidMesh;
    }

    // Build the graphics pipeline lazily from the first mesh's layout.
    if (pipeline_ == VK_NULL_HANDLE) {
        if (!createPipeline(desc.layout)) return render::kInvalidMesh;
    }

    VkMesh mesh;
    mesh.vertexBuffer = desc.vertexBuffer;
    mesh.indexBuffer  = desc.indexBuffer;
    mesh.vertexCount  = desc.vertexCount;
    mesh.indexCount   = desc.indexCount;
    mesh.indexed = desc.indexBuffer != render::kInvalidBuffer &&
                   desc.indexCount > 0 &&
                   buffers_.find(desc.indexBuffer) != buffers_.end();

    render::MeshHandle handle = nextMesh_++;
    meshes_[handle] = mesh;
    return handle;
}

void VkRenderer::destroyMesh(render::MeshHandle handle) {
    meshes_.erase(handle);  // buffers are released via destroyBuffer()
}

void VkRenderer::beginFrame(const render::FrameContext& frame) {
    clear_[0] = frame.clearColor[0];
    clear_[1] = frame.clearColor[1];
    clear_[2] = frame.clearColor[2];
    clear_[3] = frame.clearColor[3];

    if (needsResize_) {
        if (!recreateSwapchain()) return;
    }

    vkWaitForFences(device_, 1, &inFlight_[currentFrame_], VK_TRUE, UINT64_MAX);

    VkResult acq = vkAcquireNextImageKHR(device_, swapchain_, UINT64_MAX,
                                         imageAvailable_[currentFrame_],
                                         VK_NULL_HANDLE, &imageIndex_);
    if (acq == VK_ERROR_OUT_OF_DATE_KHR) {
        recreateSwapchain();
        return;  // skip this frame
    }
    if (acq != VK_SUCCESS && acq != VK_SUBOPTIMAL_KHR) {
        SDL_Log("VkRenderer: vkAcquireNextImageKHR failed (%d)", (int)acq);
        return;
    }

    vkResetFences(device_, 1, &inFlight_[currentFrame_]);

    VkCommandBuffer cmd = commandBuffers_[currentFrame_];
    vkResetCommandBuffer(cmd, 0);

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cmd, &bi);

    VkClearValue clears[2]{};
    clears[0].color = {{clear_[0], clear_[1], clear_[2], clear_[3]}};
    clears[1].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo rp{};
    rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rp.renderPass = renderPass_;
    rp.framebuffer = framebuffers_[imageIndex_];
    rp.renderArea.extent = swapExtent_;
    rp.clearValueCount = 2;
    rp.pClearValues = clears;
    vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);

    if (pipeline_ != VK_NULL_HANDLE) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);

        VkViewport vpt{};
        vpt.width = static_cast<float>(swapExtent_.width);
        vpt.height = static_cast<float>(swapExtent_.height);
        vpt.maxDepth = 1.0f;
        vkCmdSetViewport(cmd, 0, 1, &vpt);
        VkRect2D scissor{{0, 0}, swapExtent_};
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        // viewProj = clip * proj * view. The clip matrix converts GL-style NDC
        // (Y up, depth -1..1) to Vulkan (Y down, depth 0..1).
        const float clip[16] = {
            1, 0,    0,   0,
            0, -1,   0,   0,
            0, 0,  0.5f,  0,
            0, 0,  0.5f,  1,
        };
        float pv[16];
        mat4Mul(frame.proj, frame.view, pv);
        mat4Mul(clip, pv, viewProj_);
        vkCmdPushConstants(cmd, pipelineLayout_, VK_SHADER_STAGE_VERTEX_BIT, 0,
                           64, viewProj_);
    }

    frameActive_ = true;
}

void VkRenderer::submit(const render::DrawItem& item) {
    if (!frameActive_ || pipeline_ == VK_NULL_HANDLE) return;
    if (drawMode_ == 0) return;  // None: draw nothing

    auto meshIt = meshes_.find(item.mesh);
    if (meshIt == meshes_.end()) return;
    const VkMesh& mesh = meshIt->second;

    auto vbIt = buffers_.find(mesh.vertexBuffer);
    if (vbIt == buffers_.end()) return;

    VkCommandBuffer cmd = commandBuffers_[currentFrame_];

    // Common per-draw state (shared by every pipeline / drawing mode).
    VkDescriptorSet set = whiteSet_;
    auto tIt = textures_.find(item.texture);
    if (tIt != textures_.end()) set = tIt->second.set;
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout_, 0,
                            1, &set, 0, nullptr);
    vkCmdPushConstants(cmd, pipelineLayout_, VK_SHADER_STAGE_VERTEX_BIT, 64, 64, item.model);
    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &vbIt->second.buffer, &offset);
    const VkBufferObject* ib = nullptr;
    if (mesh.indexed) {
        auto ibIt = buffers_.find(mesh.indexBuffer);
        if (ibIt == buffers_.end()) return;
        ib = &ibIt->second;
        vkCmdBindIndexBuffer(cmd, ib->buffer, 0, VK_INDEX_TYPE_UINT32);
    }

    auto drawWith = [&](VkPipeline pipe, bool blackEdges) {
        float forceColor[4] = {0.0f, 0.0f, 0.0f, blackEdges ? 1.0f : 0.0f};
        vkCmdPushConstants(cmd, pipelineLayout_, VK_SHADER_STAGE_FRAGMENT_BIT, 128, 16,
                           forceColor);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          pipe != VK_NULL_HANDLE ? pipe : pipeline_);
        if (mesh.indexed)
            vkCmdDrawIndexed(cmd, mesh.indexCount, 1, 0, 0, 0);
        else
            vkCmdDraw(cmd, mesh.vertexCount, 1, 0, 0);
    };

    // Helium drawing modes (see IRenderer::setDrawMode).
    switch (drawMode_) {
        case 2:  // WireFrame
            drawWith(pipelineWireframe_, false);
            break;
        case 3:  // WireFrameOverSolid: biased fill, then black edges on top
            drawWith(pipeline_, false);
            drawWith(pipelineWireframe_, true);
            break;
        case 4:  // Point
            drawWith(pipelinePoint_, false);
            break;
        default:  // 1 = Solid
            drawWith(pipeline_, false);
            break;
    }
}

void VkRenderer::beginOverlay(const render::OverlayContext& ov) {
    if (!frameActive_ || pipeline_ == VK_NULL_HANDLE) return;
    VkCommandBuffer cmd = commandBuffers_[currentFrame_];

    // Restrict to the overlay rect (Vulkan viewport/scissor are top-left origin).
    VkViewport vp{};
    vp.x = static_cast<float>(ov.x);
    vp.y = static_cast<float>(ov.y);
    vp.width = static_cast<float>(ov.width);
    vp.height = static_cast<float>(ov.height);
    vp.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &vp);

    VkRect2D scissor{{ov.x, ov.y},
                     {static_cast<uint32_t>(ov.width), static_cast<uint32_t>(ov.height)}};
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    if (ov.clearDepth) {
        VkClearAttachment att{};
        att.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        att.clearValue.depthStencil = {1.0f, 0};
        VkClearRect cr{};
        cr.rect = scissor;
        cr.baseArrayLayer = 0;
        cr.layerCount = 1;
        vkCmdClearAttachments(cmd, 1, &att, 1, &cr);
    }

    // viewProj = clip * proj * view (same Vulkan clip correction as the scene).
    const float clip[16] = {
        1, 0,    0,   0,
        0, -1,   0,   0,
        0, 0,  0.5f,  0,
        0, 0,  0.5f,  1,
    };
    float pv[16];
    mat4Mul(ov.proj, ov.view, pv);
    mat4Mul(clip, pv, viewProj_);
    vkCmdPushConstants(cmd, pipelineLayout_, VK_SHADER_STAGE_VERTEX_BIT, 0, 64,
                       viewProj_);
}

void VkRenderer::endFrame() {
    if (!frameActive_) return;
    frameActive_ = false;

    VkCommandBuffer cmd = commandBuffers_[currentFrame_];
    vkCmdEndRenderPass(cmd);
    vkEndCommandBuffer(cmd);

    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.waitSemaphoreCount = 1;
    si.pWaitSemaphores = &imageAvailable_[currentFrame_];
    si.pWaitDstStageMask = &waitStage;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = &renderFinished_[currentFrame_];
    vkQueueSubmit(queue_, 1, &si, inFlight_[currentFrame_]);

    VkPresentInfoKHR pi{};
    pi.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = &renderFinished_[currentFrame_];
    pi.swapchainCount = 1;
    pi.pSwapchains = &swapchain_;
    pi.pImageIndices = &imageIndex_;

    VkResult pres = vkQueuePresentKHR(queue_, &pi);
    if (pres == VK_ERROR_OUT_OF_DATE_KHR || pres == VK_SUBOPTIMAL_KHR ||
        needsResize_) {
        recreateSwapchain();
    }

    currentFrame_ = (currentFrame_ + 1) % kFramesInFlight;
}

void VkRenderer::resize(uint32_t width, uint32_t height) {
    pendingW_ = width;
    pendingH_ = height;
    needsResize_ = true;
}

void VkRenderer::setVsync(bool enabled) {
    if (vsync_ == enabled) return;
    vsync_ = enabled;
    // Recreate the swapchain (with the new present mode) on the next frame.
    needsResize_ = true;
}

void VkRenderer::setDrawMode(uint32_t mode) { drawMode_ = mode; }

bool VkRenderer::readPixels(std::vector<uint8_t>&, uint32_t&, uint32_t&) {
    // TODO: copy the last swapchain image to a host-visible buffer. Screenshot
    // is currently OpenGL-only.
    SDL_Log("VkRenderer: readPixels not implemented (use the GL backend)");
    return false;
}

void VkRenderer::shutdown() {
    if (device_) vkDeviceWaitIdle(device_);

    if (gridPipeline_)   vkDestroyPipeline(device_, gridPipeline_, nullptr);
    if (gridLayout_)     vkDestroyPipelineLayout(device_, gridLayout_, nullptr);
    gridPipeline_ = VK_NULL_HANDLE;
    gridLayout_   = VK_NULL_HANDLE;
    if (pipelinePoint_)     vkDestroyPipeline(device_, pipelinePoint_, nullptr);
    pipelinePoint_ = VK_NULL_HANDLE;
    if (pipelineWireframe_) vkDestroyPipeline(device_, pipelineWireframe_, nullptr);
    pipelineWireframe_ = VK_NULL_HANDLE;
    if (pipeline_)       vkDestroyPipeline(device_, pipeline_, nullptr);
    if (pipelineLayout_) vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
    pipeline_ = VK_NULL_HANDLE;
    pipelineLayout_ = VK_NULL_HANDLE;
    meshes_.clear();

    for (auto& [handle, t] : textures_) {
        vkDestroyImageView(device_, t.view, nullptr);
        vkDestroyImage(device_, t.image, nullptr);
        vkFreeMemory(device_, t.memory, nullptr);
    }
    textures_.clear();
    if (descPool_)   vkDestroyDescriptorPool(device_, descPool_, nullptr);
    if (descLayout_) vkDestroyDescriptorSetLayout(device_, descLayout_, nullptr);
    if (sampler_)    vkDestroySampler(device_, sampler_, nullptr);
    descPool_ = VK_NULL_HANDLE;
    descLayout_ = VK_NULL_HANDLE;
    sampler_ = VK_NULL_HANDLE;

    for (auto& [handle, obj] : buffers_) {
        vkDestroyBuffer(device_, obj.buffer, nullptr);
        vkFreeMemory(device_, obj.memory, nullptr);
    }
    buffers_.clear();

    for (auto s : imageAvailable_) vkDestroySemaphore(device_, s, nullptr);
    for (auto s : renderFinished_) vkDestroySemaphore(device_, s, nullptr);
    for (auto f : inFlight_)       vkDestroyFence(device_, f, nullptr);
    imageAvailable_.clear();
    renderFinished_.clear();
    inFlight_.clear();

    if (commandPool_) vkDestroyCommandPool(device_, commandPool_, nullptr);
    commandPool_ = VK_NULL_HANDLE;

    destroySwapchain();

    if (renderPass_) vkDestroyRenderPass(device_, renderPass_, nullptr);
    renderPass_ = VK_NULL_HANDLE;

    if (device_)   vkDestroyDevice(device_, nullptr);
    device_ = VK_NULL_HANDLE;
    if (surface_)  vkDestroySurfaceKHR(instance_, surface_, nullptr);
    surface_ = VK_NULL_HANDLE;
    if (instance_) vkDestroyInstance(instance_, nullptr);
    instance_ = VK_NULL_HANDLE;
}

} // namespace orange::vk
