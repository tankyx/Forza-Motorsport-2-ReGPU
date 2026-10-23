#include "gpu/vk/vk_loader.h"
#include "gpu/vk/vk_surface.h"

#include <cstdio>

namespace vk
{
namespace
{
constexpr VkFormat kColorFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT_S8_UINT;

void Barrier(VkCommandBuffer cmd, VkImage image, VkImageAspectFlags aspect, VkImageLayout from, VkImageLayout to,
    VkAccessFlags srcAccess, VkAccessFlags dstAccess, VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage)
{
    VkImageMemoryBarrier barrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    if (from == VK_IMAGE_LAYOUT_UNDEFINED)
    {
        // Nothing happened before: no source scope (and the contents are discarded).
        srcAccess = 0;
        srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    }
    barrier.srcAccessMask = srcAccess;
    barrier.dstAccessMask = dstAccess;
    barrier.oldLayout = from;
    barrier.newLayout = to;
    barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = { aspect, 0, 1, 0, 1 };
    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

VkRenderPass MakeRenderPass(const Context& context, VkSampleCountFlagBits samples)
{
    VkAttachmentDescription attachments[2]{};
    attachments[0].format = kColorFormat;
    attachments[0].samples = samples;
    attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; // one pass per surface per frame (see the scope notes)
    attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[0].initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    attachments[0].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    attachments[1] = attachments[0];
    attachments[1].format = kDepthFormat;
    attachments[1].initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    VkAttachmentReference colorRef{ 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
    VkAttachmentReference depthRef{ 1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorRef;
    subpass.pDepthStencilAttachment = &depthRef;
    VkRenderPassCreateInfo info{ VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO };
    info.attachmentCount = 2;
    info.pAttachments = attachments;
    info.subpassCount = 1;
    info.pSubpasses = &subpass;
    VkRenderPass pass = VK_NULL_HANDLE;
    return vkCreateRenderPass(context.device, &info, nullptr, &pass) == VK_SUCCESS ? pass : VK_NULL_HANDLE;
}
}

void SurfaceCacheDestroy(Context& context, SurfaceCache& cache)
{
    for (auto& [base, surface] : cache.surfaces)
    {
        if (surface.framebuffer)
            vkDestroyFramebuffer(context.device, surface.framebuffer, nullptr);
        if (surface.pass)
            vkDestroyRenderPass(context.device, surface.pass, nullptr);
        DestroyImage(context, surface.color);
        DestroyImage(context, surface.depth);
    }
    cache.surfaces.clear();
}

Surface* SurfaceAcquire(Context& context, SurfaceCache& cache, uint32_t base, uint32_t width, uint32_t height,
    VkSampleCountFlagBits samples)
{
    auto& surface = cache.surfaces[base];
    // The guest can point the same EDRAM base at a different resolution (splitscreen, a resize),
    // and the settings can change the sample count: rebuild when either disagrees.
    if (surface.color.image && surface.width == width && surface.height == height && surface.samples == samples)
        return &surface;

    if (surface.framebuffer)
        vkDestroyFramebuffer(context.device, surface.framebuffer, nullptr);
    if (surface.pass)
        vkDestroyRenderPass(context.device, surface.pass, nullptr);
    DestroyImage(context, surface.color);
    DestroyImage(context, surface.depth);
    surface = {};
    surface.base = base;
    surface.width = width;
    surface.height = height;
    surface.samples = samples;
    surface.color = CreateImage(context, width, height, kColorFormat,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        VK_IMAGE_VIEW_TYPE_2D, 1, samples);
    surface.depth = CreateImage(context, width, height, kDepthFormat,
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_VIEW_TYPE_2D, 1, samples);
    if (!surface.color.image || !surface.depth.image)
        return nullptr;
    surface.pass = MakeRenderPass(context, samples);
    if (!surface.pass)
        return nullptr;
    VkFramebufferCreateInfo info{ VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO };
    info.renderPass = surface.pass;
    VkImageView views[2] = { surface.color.view, surface.depth.view };
    info.attachmentCount = 2;
    info.pAttachments = views;
    info.width = width;
    info.height = height;
    info.layers = 1;
    if (vkCreateFramebuffer(context.device, &info, nullptr, &surface.framebuffer) != VK_SUCCESS)
        return nullptr;
    return &surface;
}

void SurfaceBegin(Context& context, Surface& surface, VkCommandBuffer cmd, const VkClearColorValue* clearColor)
{
    Barrier(cmd, surface.color.image, VK_IMAGE_ASPECT_COLOR_BIT, surface.colorLayout, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT,
        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
    surface.colorLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    Barrier(cmd, surface.depth.image, VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, surface.depthLayout,
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT);
    surface.depthLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkRenderPassBeginInfo begin{ VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
    begin.renderPass = surface.pass;
    begin.framebuffer = surface.framebuffer;
    begin.renderArea = { { 0, 0 }, { surface.width, surface.height } };
    VkClearValue values[2]{};
    if (clearColor)
        values[0].color = *clearColor;
    // The guest renders reverse-Z (near 1, far 0) and compares GREATER, so a depth buffer cleared
    // to 1.0 is "everything already at the near plane" and rejects every fragment — which is what
    // put the car in a band at the top. Cleared to 0, the far plane, as reverse-Z wants.
    values[1].depthStencil = { 0.0f, 0 };
    begin.clearValueCount = 2;
    begin.pClearValues = values;
    vkCmdBeginRenderPass(cmd, &begin, VK_SUBPASS_CONTENTS_INLINE);
    // hlsl.cpp emits `1 - p.y / size.y * 2`, i.e. y-down like the guest's framebuffer. Vulkan's
    // NDC has y pointing down too but its framebuffer origin is top-left, so the viewport is
    // flipped to land the image the right way up (the menu text is the quickest tell).
    VkViewport viewport{ 0, float(surface.height), float(surface.width), -float(surface.height), 0, 1 };
    VkRect2D scissor{ { 0, 0 }, { surface.width, surface.height } };
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &scissor);
    (void)context;
}

void SurfaceEnd(Context& context, Surface& surface, VkCommandBuffer cmd)
{
    vkCmdEndRenderPass(cmd);
    Barrier(cmd, surface.color.image, VK_IMAGE_ASPECT_COLOR_BIT, surface.colorLayout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT);
    surface.colorLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    (void)context;
}

void SurfaceShaderRead(Context& context, Surface& surface, VkCommandBuffer cmd)
{
    vkCmdEndRenderPass(cmd);
    Barrier(cmd, surface.color.image, VK_IMAGE_ASPECT_COLOR_BIT, surface.colorLayout, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
    surface.colorLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    (void)context;
}

bool SurfaceResolve(Context& context, Surface& surface, VkCommandBuffer cmd, VkImage destination, uint32_t width,
    uint32_t height)
{
    if (width != surface.width || height != surface.height)
        return false;
    {
        // Both ends are already in TRANSFER_SRC / need TRANSFER_DST: the caller leaves the
        // surface in TRANSFER_SRC (SurfaceEnd) and the destination untouched.
        VkImageMemoryBarrier dst{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        dst.srcAccessMask = 0;
        dst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        dst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        dst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        dst.srcQueueFamilyIndex = dst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        dst.image = destination;
        dst.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &dst);
        if (surface.samples == VK_SAMPLE_COUNT_1_BIT)
        {
            VkImageCopy region{};
            region.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
            region.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
            region.extent = { width, height, 1 };
            vkCmdCopyImage(cmd, surface.color.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, destination,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        }
        else
        {
            VkImageResolve region{};
            region.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
            region.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
            region.extent = { width, height, 1 };
            vkCmdResolveImage(cmd, surface.color.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, destination,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        }
        VkImageMemoryBarrier read{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        read.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        read.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        read.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        read.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        read.srcQueueFamilyIndex = read.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        read.image = destination;
        read.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &read);
    }
    (void)context;
    return true;
}
}
