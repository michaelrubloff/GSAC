# Industry and Technical-Feasibility Context: Radiance Fields / Gaussian Splatting in an In-Car Driving Visualization (Tesla FSD "Reality View")

Research date: 2026-11-22. Method note: in this session WebFetch could not reach arxiv.org, teskas.net or kucoin.com (DNS blocked by the sandbox network policy), so the findings below come from web-search result summaries and snippets, not full-page reads. Where a figure appears only in one snippet it is flagged. Items marked "[prior knowledge, URL not re-verified this session]" come from well-known pre-2026 publications whose canonical URLs I'm confident of but could not open here.

### Context: what Tesla has said about Reality View (needed to frame every question below)
- FSD v15 began rolling out to US early testers around 2026-11-21, adding "Reality View". Coverage agrees it is AI4 (HW4) only; HW3 cars are left out. — [Paul Tan, 2026-11-22](https://paultan.org/2026/11/22/tesla-fsd-v15-rollout/); [Startup Fortune](https://startupfortune.com/tesla-starts-fsd-v15-rollout-to-early-testers-and-leaves-hw3-cars-out/); [eletric-vehicles.com](https://eletric-vehicles.com/tesla/tesla-starts-fsd-v15-rollout-adding-reality-view-and-speed-camera-slowdown/)
- Elon Musk described Reality View as "real-time AI video generation (no gaming engine!)" that uses half of one of the AI4 computer's TRIP accelerators. In a follow-up he said the computer "re-renders reality" with chevrons in the sky for turns, like a video game. — [news18a, Nov 2026](https://english.news18a.com/news/english_316936.html); [KuCoin flash news](https://www.kucoin.com/news/flash/tesla-fsd-v15-uses-half-of-ai4-chip-s-trip-accelerator-for-real-time-ai-video) (title only seen)
- Tesla's release notes reportedly describe the feature as replacing the explicit model signals with "a single end-to-end visualization using less total compute". A lower-authority site says Ashok Elluswamy stated the same model that drives also renders the scene. — [agentspulse.github.io](https://agentspulse.github.io/tutorials/tesla-fsd-v15/) (low authority; not confirmed elsewhere)
- Claims that Reality View is 3D Gaussian Splatting circulate on blogs/forums. One Japanese-language post attributes to Tesla a "220 ms vs 30 minutes for conventional Gaussian splatting" figure. A later X post argues the opposite: features are forked into a ~1B-parameter diffusion model that "paints" the view, and Tesla's generative Gaussian splatting (shown at ICCV) is separate. — [teskas.net forum thread](https://teskas.net/t/fsd-v15-ui-gta/44161); [X post by @tslaming](https://x.com/tslaming/status/2124474912161087636). I found no primary Tesla source for either the 220 ms figure or the 1B-parameter diffusion claim. Both are UNCONFIRMED.
- At ICCV 2025 (WDFM-AD workshop, Honolulu, October 2025), Ashok Elluswamy showed Tesla's learned "neural world simulator", which generates all 8 camera feeds, and showed Gaussian-splat novel-view renderings (described as a debugging/visualization tool). Coverage does not tie splats to the world simulator itself. — [Elluswamy on X, Oct 2025](https://x.com/aelluswamy/status/1981644831790379245); [NextBigFuture, Oct 2025](https://www.nextbigfuture.com/2025/10/tesla-vp-ashok-describes-technology-of-fsd.html); [Medium summary](https://medium.com/@byaman019/a-peek-into-teslas-autonomous-future-core-tech-revealed-by-vp-ashok-elluswamy-at-iccv-2025-906df7ae95de)

## 1. Which AV companies publicly use 3DGS/NeRF (and for what)? Has any automaker shipped a radiance-field/splat visualization in the car UI?

### Takeaway
Radiance fields and splats are now standard in AV **offline simulation and data tooling** (Waymo, NVIDIA, Wayve, Waabi, Xiaomi, Li Auto, Tesla's own debugging tools). Since 2025 the frontier has moved toward **generative video world models** (Waymo World Model on Genie 3, NVIDIA Cosmos/OmniDreams, Wayve GAIA-2, Tesla's neural world simulator). I found no evidence that any automaker had shipped a splat- or NeRF-rendered driver visualization to customers before Tesla's v15. Xiaomi's May 2026 statement that its Gaussian-based world model serves "smart cabin systems" is the closest, and it's vague.

### Cited Findings
**Waymo**
- Block-NeRF (CVPR 2022): a city-scale NeRF of San Francisco built from 2.8M Waymo images, for simulation/mapping. — [arXiv 2202.05263](https://arxiv.org/abs/2202.05263) [prior knowledge, URL not re-verified this session]
- EmerNeRF (ICLR 2024; Waymo, USC, Stanford, NVIDIA collaborators): self-supervised static/dynamic/flow decomposition of driving scenes. — [arXiv 2311.02077](https://arxiv.org/abs/2311.02077) [prior knowledge, URL not re-verified this session]
- SceneDiffuser (NeurIPS 2024) and SceneDiffuser++ (CVPR 2025) are diffusion models for agent/scene initialization and rollout in simulation, not radiance fields. — [Awesome-World-Model list](https://github.com/LMD0311/Awesome-World-Model) (venues conflict across lists)
- STORM (Jan 2025; Waymo-affiliated authors per prior knowledge) is a feed-forward 3DGS spatio-temporal reconstruction model with 0.18 s inference on an A100. — [arXiv 2501.00602](https://arxiv.org/pdf/2501.00602)
- "The Waymo World Model" (blog, Feb 2026) is built on Google DeepMind's Genie 3 and generates camera **and lidar** data, controllable by driving actions, scene layout and language. Waymo calls it "efficient real-time" and says a lighter variant simulates longer scenes at much lower compute. — [Waymo blog, Feb 2026](https://waymo.com/blog/2026/02/the-waymo-world-model-a-new-frontier-for-autonomous-driving-simulation)

**NVIDIA**
- 3DGRT (ray-traced Gaussians, SIGGRAPH Asia 2024) and 3DGUT (unscented-transform Gaussians that support distorted/rolling-shutter cameras, CVPR 2025) are the rendering bases of NVIDIA's Omniverse NuRec neural reconstruction for AV simulation. NuRec was productized in 2025. — [3DGRT arXiv 2407.07090](https://arxiv.org/abs/2407.07090); [3DGUT arXiv 2412.12507](https://arxiv.org/abs/2412.12507) [prior knowledge, URLs not re-verified this session]
- Instant NuRec (NVIDIA, arXiv 2607.14203, July 2026) is a feed-forward model that turns posed multi-camera frames into a layered 3DGS world (static, dynamic, sky). A 10–20 s multi-camera clip reconstructs in about 1.5 s, versus about 75 min for per-scene-optimized NuRec. PSNR on the Waymo Open Dataset is +2.01 dB over the best baseline. It was trained on about 40K clips. With LiDAR, reconstruction goes from about 1 s to about 20 s. — [NVIDIA project page](https://research.nvidia.com/labs/sil/projects/instant-nurec/); [arXiv 2607.14203](https://arxiv.org/pdf/2607.14203)
- An NVIDIA researcher says Instant NuRec "is now the official NVIDIA Omniverse NuRec product", with code and models released. — [Jiahui Huang on X, 2026](https://x.com/huangjh_hjh/status/2080035580935893180)
- OmniDreams (NVIDIA, 2026-06-02) is a generative world model trained from Cosmos diffusion that autoregressively generates action-conditioned driving video "in real time". It was trained on 21k hours of driving and runs closed-loop with the Alpamayo 1 policy and AlpaSim. Weights are open. No FPS figure was found. — [NVIDIA OmniDreams page](https://research.nvidia.com/labs/sil/projects/omnidreams-blog/); [arXiv 2606.03159](https://arxiv.org/pdf/2606.03159)

**Wayve**
- PRISM-1 (July 2024) is a custom 4D Gaussian Splatting reconstruction for Wayve's Ghost Gym simulator. It needs no labels, scene graphs or bounding boxes. The WayveScenes101 dataset (101 scenes) was released with it. — [radiancefields.com Wayve page](https://radiancefields.com/platforms/wayve) (aggregator); [Wayve PRISM-1 post](https://wayve.ai/thinking/prism-1/) [prior knowledge, URL not re-verified this session]
- GAIA-2 (arXiv, March 2025) is a controllable **multi-view generative video** world model, not 3DGS. — [Awesome-World-Model list](https://github.com/LMD0311/Awesome-World-Model)

**Waabi**
- UniSim (CVPR 2023) is a neural closed-loop sensor simulator that reconstructs camera and LiDAR from a recorded drive. — [arXiv 2308.01898](https://arxiv.org/abs/2308.01898) [prior knowledge, URL not re-verified this session]. My searches found no 2025–2026 Waabi-specific splat news (gap).

**Chinese OEMs**
- Xiaomi Auto World Model (announced 2026-05-26) pairs WorldRec, a feed-forward reconstruction into "compact yet high-fidelity 3D Gaussian scene representations" (about 10 s per reconstruction per Xiaomi's project page), with WorldGen video generation. Xiaomi says it is used in synthetic data, closed-loop simulation **and "smart cabin systems"**. World Model 2.0 (arXiv 2611.01677, Nov 2026) builds static and dynamic Gaussians in one forward pass. — [CnEVPost, 2026-05-26](https://cnevpost.com/2026/05/26/xiaomi-ev-introduces-world-model/); [arXiv 2605.18137](https://arxiv.org/html/2605.18137v2); [project page](https://jointwm.github.io/); [arXiv 2611.01677](https://arxiv.org/pdf/2611.01677)
- Li Auto's driving stack uses static BEV, dynamic BEV and occupancy networks, and reportedly uses NeRF to enhance the occupancy network (perception/training, not the display). — [Baidu wiki, Li Auto](https://baike.baidu.com/en/item/Li%20Auto/986389) (low-authority source)
- XPeng and Huawei co-developed an AR-HUD (87-inch virtual image), debuting on the XPeng G7 in June 2025. It is a conventional display stack, not reconstruction. — [CnEVPost, 2025-06-05](https://cnevpost.com/2025/06/05/xpeng-unveils-hud-system-huawei/)
- Searches for a shipped Chinese-EV "3D reconstruction" display built from splats/NeRF found nothing (see Gaps).

**Tesla itself**
- Tesla showed Gaussian-splat novel-view renderings at ICCV 2025 (Oct 2025) and a learned neural world simulator that generates 8 synthetic camera feeds for closed-loop evaluation and RL. — [Elluswamy on X](https://x.com/aelluswamy/status/1981644831790379245); [NextBigFuture](https://www.nextbigfuture.com/2025/10/tesla-vp-ashok-describes-technology-of-fsd.html)

### Inferences
- Peers use splats mostly for **offline sim and replay**, and are adding generative video models for novel or counterfactual content. Tesla has publicly shown both capabilities. So "Tesla uses splats" and "Tesla uses a video generator" are each plausible from Tesla's track record. Musk's own wording ("AI video generation, no gaming engine") points to the generative/learned-renderer side, not an explicit 3DGS rasterizer.
- If Reality View is a learned renderer, Tesla would likely be the first OEM to ship a **neural-rendered** driver visualization to customers. Whether it is technically "radiance fields" is unconfirmed. (Speculation, based on absence of contrary evidence.)

### Gaps
- No full-text confirmation of Zoox, Mobileye, Huawei ADS, NIO, Applied Intuition or Parallel Domain 3DGS/NeRF use in 2025–2026. My searches didn't surface them, and they weren't researched further within the tool budget. (Applied Intuition and Parallel Domain are widely reported to offer neural/3DGS sim products, but I have no 2025–2026 source here.)
- I could not determine what Xiaomi's "smart cabin" use of the world model actually is, or whether any Gaussian rendering reaches a customer display.
- I couldn't verify whether any Chinese EV (e.g., "real-scene" parking visualizations) uses camera-textured reconstruction versus conventional 360° bowl projection.

## 2. State of the art in feed-forward / generative Gaussian splatting for driving scenes. Is sub-second multi-camera reconstruction from 8 surround cameras realistic in 2026?

### Takeaway
Yes, on a datacenter GPU. Feed-forward driving 3DGS models reconstruct multi-camera scenes in **0.18–1.5 s on an A100-class GPU** (STORM 0.18 s; DGGT 0.39 s; Instant NuRec about 1 s camera-only / 1.5 s per 10–20 s clip). Per-scene optimization takes tens of minutes (about 75 min for NuRec). Quality is around 23–28 dB PSNR on Waymo, below per-scene optimization. Doing this continuously and in real time on a car's embedded accelerator is unproven in the literature.

### Cited Findings
- DrivingForward (AAAI 2025) predicts Gaussians per image from surround-view rigs (nuScenes 6 cameras) with a pose network and depth network. It is self-supervised (no depth GT or extrinsics in training) and claims real-time-capable feed-forward reconstruction. — [AAAI proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/32793); [project page](https://fangzhou2000.github.io/projects/drivingforward/); [GitHub](https://github.com/fangzhou2000/DrivingForward)
- STORM (arXiv 2501.00602, Jan 2025) is a feed-forward spatio-temporal 3DGS model for large outdoor scenes. It runs **0.18 s inference on a single A100** with real-time rendering, reporting 26.38 dB PSNR on Waymo in its own paper (24.48 dB in FreeForward4D's re-evaluation). — [arXiv 2501.00602](https://arxiv.org/pdf/2501.00602); [FreeForward4D arXiv 2611.24007](https://arxiv.org/pdf/2611.24007)
- DGGT (CVPR 2026) does feed-forward 4D reconstruction of dynamic driving scenes from **unposed** images in about **0.39 s per scene** on an A100 (vs 0.18 s for STORM). It's unclear whether this includes its diffusion refinement. — [CVPR 2026 open access](https://openaccess.thecvf.com/content/CVPR2026/papers/Chen_DGGT_Feedforward_4D_Reconstruction_of_Dynamic_Driving_Scenes_using_Unposed_CVPR_2026_paper.pdf); [alphaXiv](https://www.alphaxiv.org/abs/2512.03004)
- EVolSplat (arXiv 2503.20168, March 2025) is volume-based feed-forward urban 3DGS, with zero-shot Waymo PSNR 23.43 dB. EVolSplat4D (Jan 2026) extends it to 4D. — [arXiv 2503.20168](https://arxiv.org/pdf/2503.20168); [arXiv 2601.15951](https://arxiv.org/html/2601.15951v1)
- FreeForward4D (arXiv 2611.24007, Nov 2026) is annotation-free feed-forward 4D, with 27.66 dB PSNR / 0.836 SSIM on Waymo. — [arXiv 2611.24007](https://arxiv.org/pdf/2611.24007)
- Instant NuRec (NVIDIA, July 2026): about 1.5 s for a 10–20 s multi-camera clip vs about 75 min for optimized NuRec, +2.01 dB PSNR over the best feed-forward baseline on Waymo, about 1 s camera-only and about 20 s with LiDAR. — [NVIDIA project page](https://research.nvidia.com/labs/sil/projects/instant-nurec/); [arXiv 2607.14203](https://arxiv.org/pdf/2607.14203)
- Xiaomi WorldRec: feed-forward Gaussians, about 10 s per reconstruction (company claim). — [jointwm.github.io](https://jointwm.github.io/)
- Other 2025–2026 entries: DrivingScene (online feed-forward 4D, arXiv 2510.24734), XYZCylinder (cylinder lifting for cross-rig compatibility, arXiv 2510.07856), MCSplat (multi-view consistency, arXiv 2610.25345), PointForward (arXiv 2605.11594), ReconDrive (arXiv 2603.07552), ParkGaussian (surround-view 3DGS for parking, arXiv 2601.01386), GS-Occ3D (vision-only occupancy via 3DGS, arXiv 2507.19451). — [DrivingScene](https://arxiv.org/pdf/2510.24734); [XYZCylinder](https://arxiv.org/pdf/2510.07856); [MCSplat](https://arxiv.org/html/2610.25345); [PointForward](https://arxiv.org/html/2605.11594); [ParkGaussian](https://arxiv.org/pdf/2601.01386); [GS-Occ3D](https://arxiv.org/pdf/2507.19451) (details beyond titles and abstracts not verified)
- MCSplat notes a known limitation of this whole family: driving data gives dense views along the ego path but sparse views off-trajectory, so novel views far from the cameras degrade. — [arXiv 2610.25345](https://arxiv.org/html/2610.25345)

### Inferences
- **Sub-second, 8-camera feed-forward reconstruction is realistic in 2026 on an A100/H100** (STORM 0.18 s, DGGT 0.39 s). A "220 ms" per-scene number (the unverified Reality View claim) would sit right in the STORM range on datacenter hardware. On an automotive NPU it would be remarkable.
- An in-car visualization must update continuously (e.g., ≥10 Hz) while the car moves, not reconstruct one clip. That is an **online/streaming** problem (DrivingScene-style), which is less mature than per-clip feed-forward.
- Most published models assume known poses and A100-scale memory and compute. Tesla has camera calibration and ego-motion from its own stack, which removes the pose problem. (Speculation.)
- The off-trajectory sparsity limit matters for a third-person "chase cam" view like Reality View: geometry behind or above the car, and the car's own body, are never observed by the cameras and must be hallucinated or replaced with a stored asset.

### Gaps
- No found timing for Omni-Scene, DrivingRecon or SCube (SCube is NVIDIA's voxel-to-Gaussian large-scene model, about 20 s per scene from my memory; not verified this session).
- No published benchmark of any driving feed-forward 3DGS model on an automotive SoC (Orin/Thor, Tesla AI4).
- I could not read full papers to confirm camera counts and resolutions used in timing (most use 3–6 cameras at reduced resolution).

## 3. Rendering 3DGS on embedded/automotive hardware. Could Reality View render splats in real time? Could reconstruction run on the FSD computer?

### Takeaway
**Rendering** splats in real time is well within the reach of Tesla's AMD infotainment GPU. Phone GPUs already hit 74–127 FPS with optimized 3DGS, and the RDNA2 Navi 23 in newer S/X is roughly an RX 6600-class desktop part. **Reconstruction** is the hard part. Published feed-forward models need 0.2–1.5 s on an A100, and AI4's NPU throughput is publicly uncertain. Per Musk, though, Reality View runs on **half of one TRIP accelerator on the AI4 FSD computer**, not on the infotainment GPU and not in the cloud. That makes a cloud pipeline unlikely and implies a compact, compute-cheap learned model.

### Cited Findings
**Tesla hardware**
- Newer Tesla infotainment (MCU3) uses an AMD Ryzen Embedded (Zen+) APU. Model S/X (2021+) add a discrete Navi 23 RDNA2 GPU (28 CUs / 1792 shaders reported), performance "similar to AMD Pro W6600 / RX 6600". — [VideoCardz](https://videocardz.com/newz/tesla-car-computer-features-zen-ryzen-embedded-apu-and-discrete-navi-23-gpu); [Jan-Eric Schäfrich HW4 teardown notes](https://www.janschafrich.com/tesla-car-computer-hw4/)
- The "up to 10 TFLOPS, PS5-comparable" figure comes from an enthusiast post. That post also says Model 3/Y rely on the APU's integrated GPU, not a discrete GPU. I found no official spec. — [gojira GPU on X](https://x.com/gojira_gpu/status/1919405728638677237). (Note: AMD's RX 6600 spec is about 8.9 TFLOPS FP32 [prior knowledge].) Older Teslas used Intel Atom MCU2 infotainment [prior knowledge; see [TeslaTap](https://teslatap.com/articles/autopilot-processors-and-hardware-mcu-hw-demystified/)].
- AI4 FSD computer: reported 3 NN processors (vs 2 in HW3), 20 CPU cores, 16 GB RAM, 256 GB storage, about 2–3× HW3 inference. Per-chip TOPS figures conflict (a blog says 50 TOPS; a Chinese wiki says 720 TOPS). — [blogordie.com](https://www.blogordie.com/2023/09/hw4-tesla-new-self-driving-hardware/); [Bitauto wiki](https://www.bitauto.com/wiki/100135463836/); [autoedgeview](https://autoedgeview.com/ev-reviews/tesla-hw4-years-model-guide). HW3 was officially 72 TOPS per chip / 144 TOPS per board (Tesla Autonomy Day, April 2019) [prior knowledge]. Tesla has never published an official AI4 TOPS number that I could find.
- Musk: Reality View uses "half of one of the AI4 computer's TRIP accelerators". — [news18a](https://english.news18a.com/news/english_316936.html)

**Splat rendering on mobile/embedded**
- Mobile-GS (arXiv 2603.11531, 2026): about 116–127 FPS on a Snapdragon 8 Gen 3 phone GPU (1600×1063), 74 FPS sustained under thermal load, 4.6 MB model. Adapted baselines on the same phone: quantized 3DGS 8 FPS, Mini-Splatting 12, SortFreeGS 24, LocoGS-S 17. Over 1000 FPS on an RTX 3090. — [Mobile-GS project page](https://xiaobiaodu.github.io/mobile-gs-project/); [alphaXiv](https://www.alphaxiv.org/abs/2603.11531)
- Flux-GS (arXiv 2606.30017) uses WebGL and claims higher mobile FPS than Mobile-GS on the same chip (exact numbers not retrieved). — [arXiv 2606.30017](https://arxiv.org/pdf/2606.30017)
- Mobile-4DGS (arXiv 2610.05289) targets unified static plus dynamic real-time mobile Gaussian splatting. — [arXiv 2610.05289](https://arxiv.org/pdf/2610.05289) (details not verified)

### Inferences
- **Rendering:** A Navi 23 (about 9 TFLOPS) is several times an Adreno 750 phone GPU, so rendering a driving-scale splat scene (≈0.5–2M Gaussians) at 30–60 FPS on S/X infotainment is very plausible. Model 3/Y's integrated Ryzen GPU (Vega-class, ≈1–2 TFLOPS [prior knowledge, unverified]) is more marginal but feasible with compact/sort-free renderers. (Engineering estimate, speculative.)
- **Where it runs:** Musk places the work on the AI4 TRIP accelerator. That is a matrix-multiply NN engine, not a rasterizer. This is more consistent with a **neural network that directly outputs pixels** (video generation / learned decoder) than with a 3DGS rasterizer, whose sorting and alpha blending suit GPUs. A hybrid is possible: the NPU predicts Gaussians and the infotainment GPU rasterizes them. That would contradict "no gaming engine" only semantically. (Speculation.)
- **Cloud:** Musk's statement, plus 0.2–1.5 s latency, plus cellular bandwidth for 8 camera streams, make a cloud loop implausible for a live view. (Inference.)
- **On-FSD-computer reconstruction:** A STORM-class model (0.18 s on an A100 with ~312 dense FP16 tensor TFLOPS [prior knowledge]) would need heavy distillation and quantization to run at ≥10 Hz on a fraction of one AI4 NPU. The half-of-one-TRIP budget suggests a small model, possibly reusing FSD backbone features, as the low-authority "same model that drives also renders" claim implies. (Speculation.)
- **HW3 exclusion** is consistent with the compute budget: no spare NPU capacity on HW3. Also note v15 as a whole is AI4-only, so this isn't specific evidence about Reality View.

### Gaps
- No official AI4 TOPS / TRIP specification. The "half a TRIP" budget can't be converted to TOPS reliably.
- No public info on Reality View output resolution, frame rate or latency.
- Unconfirmed whether Reality View appears on Model 3/Y (integrated GPU) identically to S/X.

## 4. Alternative techniques that would look similar, and their visual tells

### Takeaway
Four families could produce a "photoreal re-rendered surroundings" look: (a) projecting camera pixels onto depth/occupancy meshes, (b) textured voxel occupancy, (c) neural textures/meshes or feed-forward Gaussians, and (d) 2D/multi-view video diffusion. Each leaves different artifacts. Musk's "AI video generation" description and the reported "diffusion painter" claim both point to (d), possibly conditioned on 3D structure.

### Cited Findings
- Tesla's FSD stack has used a vision-only occupancy network (voxels with semantics and flow) since 2022, and the pre-v15 visualization rendered occupancy/objects as stylized geometry. — [Tesla AI Day 2022 (YouTube)](https://www.youtube.com/watch?v=ODSJsviD_SU) [prior knowledge, URL not re-verified this session]
- Occupancy can itself be reconstructed or supervised via Gaussian splatting (GS-Occ3D, 2025), which blurs the line between "occupancy" and "splats". — [arXiv 2507.19451](https://arxiv.org/pdf/2507.19451)
- Generative world models (Waymo World Model on Genie 3; NVIDIA OmniDreams from Cosmos; Wayve GAIA-2; Tesla's neural world simulator) generate driving video directly and can be action-conditioned in (near) real time. — [Waymo blog, Feb 2026](https://waymo.com/blog/2026/02/the-waymo-world-model-a-new-frontier-for-autonomous-driving-simulation); [NVIDIA OmniDreams](https://research.nvidia.com/labs/sil/projects/omnidreams-blog/); [Elluswamy on X](https://x.com/aelluswamy/status/1981644831790379245)
- Hybrid reconstruction-plus-generation (Xiaomi WorldRec + WorldGen; DGGT with diffusion refinement) is an active 2026 pattern: Gaussians for geometric consistency, diffusion for filling unseen regions and removing artifacts. — [arXiv 2605.18137](https://arxiv.org/html/2605.18137v2); [DGGT CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Chen_DGGT_Feedforward_4D_Reconstruction_of_Dynamic_Driving_Scenes_using_Unposed_CVPR_2026_paper.pdf)

### Inferences (diagnostic "tells"; analyst reasoning, not sourced measurements)
- **(a) Camera projection onto depth/occupancy mesh** (like 360° "bowl" parking views). Tells: textures stretch or smear on vertical surfaces and where the mesh is wrong, tall objects "lean" or fold onto the ground, there are seams where camera frustums meet, and colors are exactly the camera's (including lens flare and exposure differences). There's no plausible content in never-seen areas (black or blurred holes behind and above the car).
- **(b) Textured voxel occupancy.** Tells: blocky, stair-stepped edges at voxel resolution (Tesla occupancy is roughly decimeter scale), thin structures (poles, wires, signs) missing or chunky, and flat per-voxel colors.
- **(c) Explicit 3DGS / neural mesh.** Tells: "floaters" and needle-shaped or blurry anisotropic blobs when the virtual camera moves away from the real cameras, view-dependent shimmer, fuzzy edges, and a geometrically rigid scene that stays perfectly consistent when you orbit. Moving objects either ghost or need separate dynamic Gaussians. Unobserved regions are empty or smeared unless filled generatively.
- **(d) 2D/multi-view video diffusion (learned painter).** Tells: no unobserved holes (everything is plausibly filled), detail that is "invented" and changes between frames (text on signs and license plates garbled or morphing), temporal flicker or "boiling" textures, objects that subtly change shape or color, a stylized consistency in lighting, and possible hallucinated objects not present in reality. Geometry may wobble slightly under camera motion rather than staying rigid. The ability to add stylized overlays (chevrons in the sky) with integrated lighting fits this approach.
- Given Musk's wording and the compute location (an NN accelerator), (d) or a (c)+(d) hybrid is the most consistent hypothesis. Pure (c) is less consistent. (Speculation.)

### Gaps
- No hands-on frame-by-frame analysis of Reality View footage was done here, so I can't say which tells are actually present.
- No Tesla technical disclosure (paper, patent or talk) specific to Reality View's architecture was found.
