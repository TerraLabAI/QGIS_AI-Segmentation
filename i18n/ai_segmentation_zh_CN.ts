<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE TS>
<TS version="2.1" language="zh_CN">
<context>
    <name>AISegmentation</name>
    <!-- Review display colours: Normal / Confidence / Random (2026-07-01) -->
    <message>
        <source>Display colors:</source>
        <translation>显示颜色：</translation>
    </message>
    <message>
        <source>Normal</source>
        <translation>普通</translation>
    </message>
    <message>
        <source>Confidence</source>
        <translation>置信度</translation>
    </message>
    <message>
        <source>Random</source>
        <translation>随机</translation>
    </message>
    <message>
        <source>Outline</source>
        <translation>轮廓</translation>
    </message>
    <message>
        <source>How detections are coloured on the map (visual only): Normal outline, Confidence heatmap (green sure, red uncertain), or a random colour per object to tell them apart.</source>
        <translation>检测结果在地图上的配色方式（仅影响显示效果）：普通轮廓、置信度热力图（绿色代表可信，红色代表不确定），或为每个对象随机分配颜色以便区分。</translation>
    </message>
    <!-- Retry: back to setup keeping inputs (2026-07-01) -->
    <message>
        <source>Retry</source>
        <translation>重试</translation>
    </message>
    <message>
        <source>Go back to your zone, references and settings to adjust and detect again. Nothing is saved.</source>
        <translation>返回您的区域、参考示例和设置进行调整，然后重新检测。系统不会保存任何内容。</translation>
    </message>
    <!-- Reference example enlarge (2026-07-01) -->
    <message>
        <source>Click to enlarge</source>
        <translation>点击放大</translation>
    </message>
    <message>
        <source>This is exactly what the AI uses: your object with a little of its surroundings.</source>
        <translation>这正是 AI 实际使用的内容：您的对象及其周边的一小部分环境。</translation>
    </message>
    <!-- Automatic step-2 redesign: describe + example + detail -->
    <message>
        <source>Describe what to find</source>
        <translation>描述要查找的内容</translation>
    </message>
    <message>
        <source>solar panel, building, tree…</source>
        <translation>solar panel, building, tree…</translation>
    </message>
    <message>
        <source>1-2 words, English</source>
        <translation>1-2 个英文单词</translation>
    </message>
    <message>
        <source>optional</source>
        <translation>可选</translation>
    </message>
    <message>
        <source>Show an example</source>
        <translation>展示示例</translation>
    </message>
    <message>
        <source>Outline one object; the AI finds the rest. No good name for it? Examples alone work too.</source>
        <translation>圈出一个对象，AI 会找到其余的。想不出合适的名称？仅凭示例也可以。</translation>
    </message>
    <message>
        <source>Exclude a look-alike</source>
        <translation>排除相似对象</translation>
    </message>
    <message>
        <source>Now outline a look-alike to exclude, then click the first point to close.</source>
        <translation>现在圈出一个要排除的相似对象，然后点击起始点以闭合。</translation>
    </message>
    <message>
        <source>Your examples drive the search.</source>
        <translation>您的示例将引导搜索。</translation>
    </message>
    <message>
        <source>Too generic to name. Clear the box to search from your example alone, or type a concrete object.</source>
        <translation>名称过于笼统。清空输入框以仅根据示例搜索，或输入一个具体的对象。</translation>
    </message>
    <message>
        <source>Example match</source>
        <translation>示例匹配</translation>
    </message>
    <message>
        <source>Include</source>
        <translation>包含</translation>
    </message>
    <message>
        <source>Exclude</source>
        <translation>排除</translation>
    </message>
    <message>
        <source>Mark an object to find more like it.</source>
        <translation>标记一个对象，以查找更多相似对象。</translation>
    </message>
    <message>
        <source>Mark a false positive to drop things like it.</source>
        <translation>标记一个误检结果，以剔除类似的结果。</translation>
    </message>
    <message>
        <source>Draw on map</source>
        <translation>在地图上绘制</translation>
    </message>
    <message>
        <source>Outline one object on the map; SAM finds all similar ones.</source>
        <translation>在地图上圈出一个对象，SAM 会找到所有相似的对象。</translation>
    </message>
    <message>
        <source>Finer detail finds smaller objects.</source>
        <translation>更精细的细节可发现更小的对象。</translation>
    </message>
    <message>
        <source>{n} object(s) detected</source>
        <translation>检测到 {n} 个对象</translation>
    </message>
    <message>
        <source>Adjust below, then export</source>
        <translation>在下方调整，然后 Export</translation>
    </message>
    <message>
        <source>Refine in Manual mode</source>
        <translation>在手动模式中细化</translation>
    </message>
    <message>
        <source>Some objects off? Refine them in Manual mode first.</source>
        <translation>有对象不准确？先在手动模式中细化它们。</translation>
    </message>
    <message>
        <source>Now outline one object on the map, then double-click to finish.</source>
        <translation>现在在地图上圈出一个对象，然后双击完成。</translation>
    </message>
    <message>
        <source>Now outline one false positive on the map, then double-click to finish.</source>
        <translation>现在在地图上圈出一个误检对象，然后双击完成。</translation>
    </message>
    <!-- Refine in Manual handoff -->
    <message>
        <source>Refine in Manual</source>
        <translation>在手动模式中细化</translation>
    </message>
    <message>
        <source>Open these detections in Manual mode to fix specific objects with point-and-click, then return here to Finish.</source>
        <translation>在手动模式中打开这些检测结果，通过点选修正特定对象，然后返回此处完成。</translation>
    </message>
    <message>
        <source>Refining Automatic results</source>
        <translation>正在细化自动检测结果</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/build.py" line="0"/>
        <source>Fine-tune the detections, then go back to review to export.</source>
        <translation>微调检测结果，然后返回复核界面进行 Export。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/state.py" line="0"/>
        <source>Editing this detection.</source>
        <translation>正在编辑此检测结果。</translation>
    </message>
    <message>
        <source>Editing this detection</source>
        <translation>正在编辑此检测结果</translation>
    </message>
    <message>
        <source>adds area</source>
        <translation>增加面积</translation>
    </message>
    <message>
        <source>removes area</source>
        <translation>减少面积</translation>
    </message>
    <message>
        <source>keeps it (turns green)</source>
        <translation>保留（变为绿色）</translation>
    </message>
    <message>
        <source>removes the object</source>
        <translation>删除该对象</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/state.py" line="0"/>
        <source>Press S to keep it (turns green) · Delete removes it</source>
        <translation>按 S 键保留（变为绿色）· 按 Delete 键删除</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/state.py" line="0"/>
        <source>Click a blue detection to open it for editing.</source>
        <translation>点击蓝色的检测结果以打开编辑。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/state.py" line="0"/>
        <source>Left-click adds area, right-click removes it. Press S to keep it (turns green).</source>
        <translation>左键点击增加面积，右键点击减少面积。按 S 键保留（变为绿色）。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/state.py" line="0"/>
        <source>{kept} of {total} kept - 'Back to review' to export.</source>
        <translation>已保留 {kept}/{total} 个，点击“返回复核”进行 Export。</translation>
    </message>
    <message>
        <source>Back to review</source>
        <translation>返回复核</translation>
    </message>
    <message>
        <source>Finish or go back to review to switch modes.</source>
        <translation>请先完成或返回复核，再切换模式。</translation>
    </message>
    <message>
        <source>Finish or exit the review to switch modes.</source>
        <translation>请先完成或退出复核，再切换模式。</translation>
    </message>
    <message>
        <source>Preparing Manual mode, loading the local model...</source>
        <translation>正在准备手动模式，加载本地模型...</translation>
    </message>
    <message>
        <source>Blue = detections to review, one at a time.</source>
        <translation>蓝色 = 待复核的检测结果，逐个处理。</translation>
    </message>
    <message>
        <source>Left-click a detection to edit it (adds area); right-click to remove a part</source>
        <translation>左键点击检测结果进行编辑（增加面积）；右键点击移除部分区域</translation>
    </message>
    <message>
        <source>Press S to validate it (turns green), then move on to the next one.</source>
        <translation>按 S 键确认（变为绿色），然后进入下一个。</translation>
    </message>
    <message>
        <source>Locked - refined in Manual mode</source>
        <translation>已锁定（已在手动模式中细化）</translation>
    </message>
    <message>
        <source>Confidence is locked while you refine in Manual mode.</source>
        <translation>在手动模式中细化时，置信度将被锁定。</translation>
    </message>
    <message>
        <source>Refining in Manual needs the local model. Open Manual mode once to finish setup, then try again.</source>
        <translation>在手动模式中细化需要本地模型。请先打开一次手动模式完成安装，然后重试。</translation>
    </message>
    <message>
        <source>Detection</source>
        <translation>检测</translation>
    </message>
    <message>
        <source>Confidence:</source>
        <translation>置信度：</translation>
    </message>
    <message>
        <source>Minimum confidence to keep a detected object. Lower finds more objects but may add false positives; raise it for cleaner results on large, distinct features.</source>
        <translation>保留检测对象所需的最低置信度。数值越低，发现的对象越多，但可能增加误检；对于大型、独立的地物，可提高数值以获得更干净的结果。</translation>
    </message>
    <!-- Account Settings: Dependencies -->
    <message>
        <source>Dependencies</source>
        <translation>依赖项</translation>
    </message>
    <message>
        <source>Local AI model files stored on this computer.</source>
        <translation>存储在本机上的本地 AI 模型文件。</translation>
    </message>
    <message>
        <source>On disk</source>
        <translation>磁盘占用</translation>
    </message>
    <message>
        <source>Not installed</source>
        <translation>未安装</translation>
    </message>
    <message>
        <source>Open folder</source>
        <translation>打开文件夹</translation>
    </message>
    <!-- v1.2 strings previously missing from translations -->
    <message>
        <source>Accept the Terms and Privacy Policy to enable segmentation.</source>
        <translation>接受服务条款和隐私政策以启用分割功能。</translation>
    </message>
    <message>
        <source>An unexpected error occurred during export. Please check the logs.</source>
        <translation>Export 过程中发生意外错误，请查看日志。</translation>
    </message>
    <message>
        <source>I agree to the &lt;a href=&quot;{terms}&quot;&gt;Terms&lt;/a&gt; and &lt;a href=&quot;{privacy}&quot;&gt;Privacy Policy&lt;/a&gt;</source>
        <translation>我同意 &lt;a href=&quot;{terms}&quot;&gt;服务条款&lt;/a&gt;和&lt;a href=&quot;{privacy}&quot;&gt;隐私政策&lt;/a&gt;</translation>
    </message>
    <message>
        <source>No valid polygons could be created from the selection. Try adjusting the refine settings or making a new selection.</source>
        <translation>无法从选区生成有效的多边形。请尝试调整细化设置或重新创建选区。</translation>
    </message>
    <message>
        <source>Repairing Installation</source>
        <translation>正在修复安装</translation>
    </message>
    <message>
        <source>Repairing installation...</source>
        <translation>正在修复安装...</translation>
    </message>
    <message>
        <source>Restart QGIS Required</source>
        <translation>需要重启 QGIS</translation>
    </message>
    <message>
        <source>Something went wrong with this click, so it was not applied. Please try again.</source>
        <translation>此次点击出现问题，未生效，请重试。</translation>
    </message>
    <message>
        <source>The Python runtime used by the AI engine is damaged (this can be caused by a disk cleanup tool or antivirus). It will now be repaired automatically. Please try your selection again once the repair finishes.</source>
        <translation>AI 引擎所使用的 Python 运行环境已损坏（可能由磁盘清理工具或防病毒软件导致），系统将自动修复。修复完成后请重新尝试选择。</translation>
    </message>
    <message>
        <source>Your polygons were added as a temporary layer so nothing is lost.</source>
        <translation>您的多边形已添加为临时图层，不会丢失。</translation>
    </message>
    <message>
        <source>Could not write to {name}. Saved to a separate file instead.</source>
        <translation>无法写入 {name}，已改为保存到其他文件。</translation>
    </message>
    <!-- Welcome Section -->
    <message>
        <source>Click Install to set up AI Segmentation</source>
        <translation>点击“安装”以配置 AI Segmentation</translation>
    </message>

    <!-- Setup Section -->
    <message>
        <source>Installing AI Segmentation...</source>
        <translation>正在安装 AI Segmentation...</translation>
    </message>
    <message>
        <source>Verifying installation...</source>
        <translation>正在验证安装...</translation>
    </message>
    <message>
        <source>Detecting device...</source>
        <translation>正在检测设备...</translation>
    </message>
    <message>
        <source>Install path: {}</source>
        <translation>安装路径：{}</translation>
    </message>
    <message>
        <source>To install in a different folder, set the environment variable AI_SEGMENTATION_CACHE_DIR:</source>
        <translation>要安装到其他文件夹，请设置环境变量 AI_SEGMENTATION_CACHE_DIR：</translation>
    </message>
    <message>
        <source>1. Open Windows Settings &gt; System &gt; Advanced system settings
2. Click 'Environment Variables'
3. Under 'User variables', click 'New'
4. Variable name: AI_SEGMENTATION_CACHE_DIR
5. Variable value: the folder path you want to use
6. Click OK and restart QGIS</source>
        <translation>1. 打开 Windows 设置 &gt; 系统 &gt; 高级系统设置
2. 点击“环境变量”
3. 在“用户变量”下，点击“新建”
4. 变量名：AI_SEGMENTATION_CACHE_DIR
5. 变量值：您希望使用的文件夹路径
6. 点击“确定”并重启 QGIS</translation>
    </message>
    <message>
        <source>Run this command in Terminal, then restart QGIS:

launchctl setenv AI_SEGMENTATION_CACHE_DIR /your/path</source>
        <translation>在终端中运行以下命令，然后重启 QGIS：

launchctl setenv AI_SEGMENTATION_CACHE_DIR /your/path</translation>
    </message>
    <message>
        <source>Add this line to your ~/.bashrc or ~/.profile, then restart QGIS:

export AI_SEGMENTATION_CACHE_DIR=/your/path</source>
        <translation>将以下这一行添加到您的 ~/.bashrc 或 ~/.profile，然后重启 QGIS：

export AI_SEGMENTATION_CACHE_DIR=/your/path</translation>
    </message>
    <message>
        <source>Checking...</source>
        <translation>正在检查...</translation>
    </message>
    <message>
        <source>Install</source>
        <translation>安装</translation>
    </message>
    <message>
        <source>Update</source>
        <translation>更新</translation>
    </message>
    <message>
        <source>Dependencies ready</source>
        <translation>依赖项已就绪</translation>
    </message>
    <message>
        <source>Downloading AI model...</source>
        <translation>正在下载 AI 模型...</translation>
    </message>
    <message>
        <source>Dependencies ready, model not downloaded</source>
        <translation>依赖项已就绪，模型尚未下载</translation>
    </message>
    <message>
        <source>Dependencies ready, model download failed</source>
        <translation>依赖项已就绪，模型下载失败</translation>
    </message>
    <message>
        <source>Download Model</source>
        <translation>下载模型</translation>
    </message>
    <message>
        <source>Cancel</source>
        <translation>取消</translation>
    </message>
    <message>
        <source>Cancel installation</source>
        <translation>取消安装</translation>
    </message>
    <message>
        <source>Are you sure you want to cancel the installation?</source>
        <translation>确定要取消安装吗？</translation>
    </message>
    <message>
        <source>Installation cancelled</source>
        <translation>安装已取消</translation>
    </message>
    <message>
        <source>Installation failed</source>
        <translation>安装失败</translation>
    </message>
    <message>
        <source>Verification failed:</source>
        <translation>验证失败：</translation>
    </message>
    <message>
        <source>Verification Failed</source>
        <translation>验证失败</translation>
    </message>
    <message>
        <source>Virtual environment was created but verification failed:</source>
        <translation>虚拟环境已创建，但验证失败：</translation>
    </message>
    <message>
        <source>Unknown error</source>
        <translation>未知错误</translation>
    </message>
    <message>
        <source>Installation Failed</source>
        <translation>安装失败</translation>
    </message>

    <!-- Model Section -->
    <message>
        <source>Update QGIS to 3.34+ for the latest AI model</source>
        <translation>请将 QGIS 更新到 3.34 及以上版本以使用最新的 AI 模型</translation>
    </message>
    <message>
        <source>Intel Mac: using SAM1 (compatible with PyTorch 2.2)</source>
        <translation>Intel Mac：使用 SAM1（兼容 PyTorch 2.2）</translation>
    </message>
    <message>
        <source>Download Failed</source>
        <translation>下载失败</translation>
    </message>
    <message>
        <source>Failed to download model:</source>
        <translation>模型下载失败：</translation>
    </message>

    <!-- Panel Title -->
    <message>
        <source>AI Segmentation by TerraLab</source>
        <translation>AI Segmentation · TerraLab 出品</translation>
    </message>

    <!-- Segmentation Section -->
    <message>
        <source>Select a Raster Layer to Segment:</source>
        <translation>选择要分割的栅格图层：</translation>
    </message>
    <message>
        <source>Select a raster layer (GeoTIFF, WMS, XYZ tiles, etc.)</source>
        <translation>选择一个栅格图层（GeoTIFF、WMS、XYZ 瓦片等）</translation>
    </message>
    <message>
        <source>No raster layer found. Add a GeoTIFF, image file, or online layer (WMS, XYZ) to your project.</source>
        <translation>未找到栅格图层。请向项目中添加 GeoTIFF、图像文件或在线图层（WMS、XYZ）。</translation>
    </message>
    <message>
        <source>No layer found. Add a raster or online layer to your project.</source>
        <translation>未找到图层。请向项目中添加栅格图层或在线图层。</translation>
    </message>
    <message>
        <source>Start AI Segmentation</source>
        <translation>开始 AI Segmentation</translation>
    </message>
    <message>
        <source>Save polygon</source>
        <translation>保存多边形</translation>
    </message>
    <message>
        <source>Undo last point</source>
        <translation>撤销上一个点</translation>
    </message>
    <message>
        <source>Stop segmentation</source>
        <translation>停止分割</translation>
    </message>
    <message>
        <source>Segmentation</source>
        <translation>分割</translation>
    </message>
    <message>
        <source>Navigation</source>
        <translation>导航</translation>
    </message>
    <message>
        <source>Space</source>
        <translation>空格键</translation>
    </message>
    <message>
        <source>Hold and move to pan the map</source>
        <translation>按住并移动以平移地图</translation>
    </message>
    <message>
        <source>Middle mouse button</source>
        <translation>鼠标中键</translation>
    </message>
    <message>
        <source>Click and drag to pan the map</source>
        <translation>点击并拖动以平移地图</translation>
    </message>
    <message>
        <source>Shortcuts</source>
        <translation>快捷键</translation>
    </message>
    <message>
        <source>Save current polygon to your session</source>
        <translation>将当前多边形保存到本次会话</translation>
    </message>
    <message>
        <source>The AI model works best on one element at a time.</source>
        <translation>AI 模型在一次处理一个对象时效果最佳。</translation>
    </message>
    <message>
        <source>Save your polygon before selecting the next element.</source>
        <translation>请先保存多边形，再选择下一个对象。</translation>
    </message>
    <message>
        <source>Export polygon to a layer</source>
        <translation>将多边形 Export 到图层</translation>
    </message>
    <message>
        <source>Export {count} polygons to a layer</source>
        <translation>将 {count} 个多边形 Export 到图层</translation>
    </message>

    <!-- Refine Section -->
    <message>
        <source>Refine selection</source>
        <translation>细化选区</translation>
    </message>
    <message>
        <source>Expand/Contract:</source>
        <translation>扩展/收缩：</translation>
    </message>
    <message>
        <source>Positive = expand outward, Negative = shrink inward</source>
        <translation>正值 = 向外扩展，负值 = 向内收缩</translation>
    </message>
    <message>
        <source>Simplify outline:</source>
        <translation>简化轮廓：</translation>
    </message>
    <message>
        <source>Reduce small variations in the outline (0 = no change)</source>
        <translation>减少轮廓上的细小起伏（0 = 不做更改）</translation>
    </message>
    <message>
        <source>Fill holes:</source>
        <translation>填充孔洞：</translation>
    </message>
    <message>
        <source>Fill interior holes in the selection</source>
        <translation>填充选区内部的孔洞</translation>
    </message>
    <message>
        <source>Min area:</source>
        <translation>最小面积：</translation>
    </message>
    <message>
        <source>Remove polygons smaller than this area (in pixels)</source>
        <translation>移除面积小于此值的多边形（单位：像素）</translation>
    </message>
    <message>
        <source>Shape</source>
        <translation>形状</translation>
    </message>
    <message>
        <source>Size</source>
        <translation>大小</translation>
    </message>
    <message>
        <source>Round corners:</source>
        <translation>圆角：</translation>
    </message>
    <message>
        <source>Round corners for natural shapes like trees and bushes. Increase 'Simplify outline' for smoother results.</source>
        <translation>为树木、灌木等自然形状做圆角处理。提高“简化轮廓”数值可获得更平滑的结果。</translation>
    </message>
    <message>
        <source>Outline</source>
        <translation>轮廓</translation>
    </message>
    <message>
        <source>Selection</source>
        <translation>选区</translation>
    </message>

    <!-- Instructions -->
    <message>
        <source>Click on the element you want to segment:</source>
        <translation>点击您要分割的对象：</translation>
    </message>
    <message>
        <source>Left-click to select</source>
        <translation>左键点击以选择</translation>
    </message>
    <message>
        <source>Left-click to add more</source>
        <translation>左键点击以添加更多</translation>
    </message>
    <message>
        <source>Right-click to exclude from selection</source>
        <translation>右键点击以从选区中排除</translation>
    </message>
    <message>
        <source>Invalid Layer</source>
        <translation>无效图层</translation>
    </message>
    <message>
        <source>Layer extent contains invalid coordinates (NaN/Inf). Check the raster file.</source>
        <translation>图层范围包含无效坐标（NaN/Inf），请检查栅格文件。</translation>
    </message>

    <!-- Dialogs -->
    <message>
        <source>Not Ready</source>
        <translation>尚未就绪</translation>
    </message>
    <message>
        <source>Please wait for the SAM model to load.</source>
        <translation>请等待 SAM 模型加载完成。</translation>
    </message>
    <message>
        <source>Load Failed</source>
        <translation>加载失败</translation>
    </message>
    <message>
        <source>Layer Creation Failed</source>
        <translation>图层创建失败</translation>
    </message>
    <message>
        <source>Could not create the output layer.</source>
        <translation>无法创建输出图层。</translation>
    </message>
    <message>
        <source>Export Failed</source>
        <translation>Export 失败</translation>
    </message>
    <message>
        <source>Could not save layer to file:</source>
        <translation>无法将图层保存到文件：</translation>
    </message>
    <message>
        <source>Layer was saved but could not be loaded:</source>
        <translation>图层已保存，但无法加载：</translation>
    </message>
    <message>
        <source>You have {count} unsaved polygon(s).</source>
        <translation>您有 {count} 个未保存的多边形。</translation>
    </message>
    <message>
        <source>Changing layer will discard your current segmentation. Continue?</source>
        <translation>切换图层将丢弃当前的分割结果，是否继续？</translation>
    </message>
    <message>
        <source>Change Layer?</source>
        <translation>切换图层？</translation>
    </message>
    <message>
        <source>Stop Segmentation?</source>
        <translation>停止分割？</translation>
    </message>
    <message>
        <source>This will discard {count} polygon(s).</source>
        <translation>这将丢弃 {count} 个多边形。</translation>
    </message>
    <message>
        <source>Use 'Export to layer' to keep them.</source>
        <translation>使用“Export 到图层”以保留它们。</translation>
    </message>
    <message>
        <source>This will end the current segmentation session.</source>
        <translation>这将结束当前的分割会话。</translation>
    </message>
    <message>
        <source>Do you want to continue?</source>
        <translation>是否继续？</translation>
    </message>
    <message>
        <source>Edit saved polygon</source>
        <translation>编辑已保存的多边形</translation>
    </message>
    <message>
        <source>Warning: you are about to edit an already saved polygon.</source>
        <translation>警告：您即将编辑一个已保存的多边形。</translation>
    </message>
    <message>
        <source>New to AI Segmentation?</source>
        <translation>刚开始使用 AI Segmentation？</translation>
    </message>
    <message>
        <source>Watch our tutorial</source>
        <translation>观看我们的教程</translation>
    </message>

    <!-- About Section -->
    <message>
        <source>Contact us</source>
        <translation>联系我们</translation>
    </message>
    <message>
        <source>Bug, question, feature request?</source>
        <translation>遇到问题、有疑问或功能建议？</translation>
    </message>
    <message>
        <source>We'd love to hear from you!</source>
        <translation>我们很想听听您的反馈！</translation>
    </message>
    <message>
        <source>Copy email address</source>
        <translation>复制邮箱地址</translation>
    </message>
    <message>
        <source>or</source>
        <translation>或</translation>
    </message>
    <message>
        <source>Book a video call</source>
        <translation>预约视频通话</translation>
    </message>
    <message>
        <source>Tutorial</source>
        <translation>教程</translation>
    </message>
    <message>
        <source>Settings</source>
        <translation>设置</translation>
    </message>
    <message>
        <source>Help</source>
        <translation>帮助</translation>
    </message>
    <message>
        <source>Terms</source>
        <translation>条款</translation>
    </message>
    <message>
        <source>Privacy</source>
        <translation>隐私</translation>
    </message>
    <message>
        <source>Something not working?</source>
        <translation>遇到问题了？</translation>
    </message>
    <message>
        <source>Copy your logs and send them to us, we'll look into it :)</source>
        <translation>复制您的日志并发送给我们，我们会及时处理 :)</translation>
    </message>

    <!-- Tooltip -->
    <message>
        <source>Segment elements on raster images using AI</source>
        <translation>使用 AI 分割栅格图像中的对象</translation>
    </message>

    <!-- Error Report Dialog -->
    <message>
        <source>Copy your logs with the button below and send them to our email.</source>
        <translation>使用下方按钮复制日志，并发送到我们的邮箱。</translation>
    </message>
    <message>
        <source>We'll fix your issue :)</source>
        <translation>我们会解决您的问题 :)</translation>
    </message>
    <message>
        <source>1. Click to copy logs</source>
        <translation>1. 点击以复制日志</translation>
    </message>
    <message>
        <source>2. Click to send to {}</source>
        <translation>2. 点击以发送到 {}</translation>
    </message>
    <message>
        <source>Open email client</source>
        <translation>打开邮件客户端</translation>
    </message>
    <message>
        <source>Copied!</source>
        <translation>已复制！</translation>
    </message>

    <!-- SSL / Antivirus error titles -->
    <message>
        <source>SSL Certificate Error</source>
        <translation>SSL 证书错误</translation>
    </message>
    <message>
        <source>Installation Blocked</source>
        <translation>安装被阻止</translation>
    </message>

    <message>
        <source>Click is outside the &apos;{layer}&apos; raster. To segment another raster, stop the current segmentation first.</source>
        <translation>点击位置超出栅格“{layer}”的范围。要分割其他栅格，请先停止当前的分割。</translation>
    </message>

    <!-- Update notification -->
    <message>
        <source>Big update dropped — v{version} is here!</source>
        <translation>重磅更新来啦，v{version} 已发布！</translation>
    </message>
    <message>
        <source>Grab it now</source>
        <translation>立即获取</translation>
    </message>

    <!-- Format conversion -->
    <message>
        <source>{ext} format is not directly supported. GDAL is not available.
Please convert your raster to GeoTIFF (.tif) before using AI Segmentation.</source>
        <translation>不直接支持 {ext} 格式，且 GDAL 不可用。
请先将栅格转换为 GeoTIFF（.tif）格式，然后再使用 AI Segmentation。</translation>
    </message>
    <message>
        <source>Cannot open {ext} file. The format may not be supported by your QGIS installation.
Please convert your raster to GeoTIFF (.tif) before using AI Segmentation.</source>
        <translation>无法打开 {ext} 文件，您的 QGIS 安装可能不支持该格式。
请先将栅格转换为 GeoTIFF（.tif）格式，然后再使用 AI Segmentation。</translation>
    </message>
    <message>
        <source>Failed to read {ext} file: {error}
Please convert your raster to GeoTIFF (.tif) manually.</source>
        <translation>读取 {ext} 文件失败：{error}
请手动将栅格转换为 GeoTIFF（.tif）格式。</translation>
    </message>

    <!-- PyTorch DLL Error Messages -->
    <message>
        <source>PyTorch Error</source>
        <translation>PyTorch 错误</translation>
    </message>
    <message>
        <source>PyTorch cannot load on Windows</source>
        <translation>PyTorch 无法在 Windows 上加载</translation>
    </message>
    <message>
        <source>The plugin requires Visual C++ Redistributables to run PyTorch.

Please download and install:
https://aka.ms/vs/17/release/vc_redist.x64.exe

After installation, restart QGIS and try again.</source>
        <translation>该插件需要 Visual C++ Redistributables 才能运行 PyTorch。

请下载并安装：
https://aka.ms/vs/17/release/vc_redist.x64.exe

安装完成后，请重启 QGIS 并重试。</translation>
    </message>
    <message>
        <source>Prediction Error</source>
        <translation>预测错误</translation>
    </message>
    <message>
        <source>Segmentation failed</source>
        <translation>分割失败</translation>
    </message>
    <message>
        <source>Layer data provider is not available.</source>
        <translation>图层数据源不可用。</translation>
    </message>
    <message>
        <source>Failed to fetch tiles from the online layer. Check your network connection.</source>
        <translation>无法从在线图层获取瓦片，请检查网络连接。</translation>
    </message>
    <message>
        <source>Online layer returned blank tiles for this area. Try panning to an area with data coverage.</source>
        <translation>在线图层在该区域返回了空白瓦片，请尝试平移到有数据覆盖的区域。</translation>
    </message>
    <message>
        <source>Crop Error</source>
        <translation>裁剪错误</translation>
    </message>
    <message>
        <source>No raster file path available. Please restart segmentation.</source>
        <translation>没有可用的栅格文件路径，请重新开始分割。</translation>
    </message>
    <message>
        <source>Encoding Error</source>
        <translation>编码错误</translation>
    </message>
    <message>
        <source>Report a Bug</source>
        <translation>报告问题</translation>
    </message>
    <message>
        <source>Disconnected parts detected in your polygon.</source>
        <translation>检测到您的多边形中存在不连续的部分。</translation>
    </message>
    <message>
        <source>For best accuracy, segment one element at a time.</source>
        <translation>为获得最佳精度，请一次只分割一个对象。</translation>
    </message>
    <message>
        <source>Layer: {}</source>
        <translation>图层：{}</translation>
    </message>
    <message>
        <source>Polygon saved! Click on another element to segment, or export your polygons.</source>
        <translation>多边形已保存！点击其他对象继续分割，或 Export 您的多边形。</translation>
    </message>
    <message>
        <source>Disconnected parts detected. For best accuracy, segment one element at a time.</source>
        <translation>检测到不连续的部分。为获得最佳精度，请一次只分割一个对象。</translation>
    </message>

    <message>
        <source>No element detected at this point. Try clicking on a different area.</source>
        <translation>在此位置未检测到对象，请尝试点击其他区域。</translation>
    </message>

    <message>
        <source>Updating...</source>
        <translation>正在更新...</translation>
    </message>

    <message>
        <source>Check for Updates</source>
        <translation>检查更新</translation>
    </message>

    <message>
        <source>More from TerraLab...</source>
        <translation>更多 TerraLab 产品...</translation>
    </message>

    <message>
        <source>Missing Visual C++ Redistributable. Install it, restart your computer, then click Retry.</source>
        <translation>缺少 Visual C++ Redistributable。请安装后重启电脑，然后点击“重试”。</translation>
    </message>

    <message>
        <source>Retry</source>
        <translation>重试</translation>
    </message>
<!-- v1.0.0 strings -->
    <message>
        <source>Cannot Write Export</source>
        <translation>无法写入 Export</translation>
    </message>
    <message>
        <source>Cannot create export directory '{path}': {reason}</source>
        <translation>无法创建 Export 目录“{path}”：{reason}</translation>
    </message>
    <message>
        <source>The export directory '{path}' is not writable. Choose a different location.</source>
        <translation>Export 目录“{path}”不可写，请选择其他位置。</translation>
    </message>
    <message>
        <source>Loading AI model...</source>
        <translation>正在加载 AI 模型...</translation>
    </message>
    <message>
        <source>SAM model ready</source>
        <translation>SAM 模型已就绪</translation>
    </message>
    <message>
        <source>Ready</source>
        <translation>就绪</translation>
    </message>
    <message>
        <source>Model load failed</source>
        <translation>模型加载失败</translation>
    </message>
    <message>
        <source>Click landed outside the current element — segment one element at a time. Saving the current selection and starting a new one.</source>
        <translation>点击位置超出当前对象的范围，请一次只分割一个对象。系统将保存当前选区并开始新的选区。</translation>
    </message>
    <message>
        <source>New here?</source>
        <translation>第一次使用？</translation>
    </message>
    <message>
        <source>Watch the tutorial</source>
        <translation>观看教程</translation>
    </message>
    <message>
        <source>Network Connection Problem</source>
        <translation>网络连接问题</translation>
    </message>
    <message>
        <source>Your connection appears unstable or blocked. Check: (1) your internet is working, (2) QGIS > Settings > Options > Network has a proxy configured if you are on a corporate network, (3) your firewall allows connections to pypi.org and files.pythonhosted.org.</source>
        <translation>您的网络连接似乎不稳定或被阻止。请检查：(1) 您的网络是否正常；(2) 如果您在公司网络环境下，QGIS > 设置 > 选项 > 网络中是否已配置代理；(3) 您的防火墙是否允许连接 pypi.org 和 files.pythonhosted.org。</translation>
    </message>
    <message>
        <source>Sign in to TerraLab</source>
        <translation>登录 TerraLab</translation>
    </message>
    <message>
        <source>Two steps to start using AI Segmentation</source>
        <translation>两步即可开始使用 AI Segmentation</translation>
    </message>
    <message>
        <source>1. Sign up or sign in on terra-lab.ai to get your key</source>
        <translation>1. 在 terra-lab.ai 上注册或登录以获取密钥</translation>
    </message>
    <message>
        <source>2. Paste your key below to activate</source>
        <translation>2. 在下方粘贴密钥以激活</translation>
    </message>
    <message>
        <source>1. Sign up / Sign in</source>
        <translation>1. 注册 / 登录</translation>
    </message>
    <message>
        <source>Get Your Key</source>
        <translation>获取您的密钥</translation>
    </message>
    <message>
        <source>2. Paste your activation key</source>
        <translation>2. 粘贴您的激活密钥</translation>
    </message>
    <message>
        <source>Sign in to get your key</source>
        <translation>登录以获取密钥</translation>
    </message>
    <message>
        <source>Create your free TerraLab account or sign in, then copy your activation key from the dashboard.</source>
        <translation>创建您的免费 TerraLab 账户或登录，然后从控制面板中复制您的激活密钥。</translation>
    </message>
    <message>
        <source>Activate</source>
        <translation>激活</translation>
    </message>
    <message>
        <source>Please enter your activation key.</source>
        <translation>请输入您的激活密钥。</translation>
    </message>
    <message>
        <source>Checking...</source>
        <translation>正在检查...</translation>
    </message>
    <message>
        <source>Activation key verified!</source>
        <translation>激活密钥验证成功！</translation>
    </message>
    <message>
        <source>Invalid activation key.</source>
        <translation>激活密钥无效。</translation>
    </message>
    <message>
        <source>Cannot reach server. Check your internet connection.</source>
        <translation>无法连接服务器，请检查您的网络连接。</translation>
    </message>
    <message>
        <source>Signed in!</source>
        <translation>已登录！</translation>
    </message>
    <message>
        <source>AI Edit</source>
        <translation>AI Edit</translation>
    </message>
    <message>
        <source>Generate imagery with AI on map zones (opens AI Edit plugin)</source>
        <translation>使用 AI 为地图区域生成图像（打开 AI Edit 插件）</translation>
    </message>
    <message>
        <source>Right-click must be inside the current selection area.</source>
        <translation>右键点击必须在当前选区范围内。</translation>
    </message>
    <!-- Account Settings Dialog -->
    <message>
        <source>Account Settings</source>
        <translation>账户设置</translation>
    </message>
    <message>
        <source>Loading account info...</source>
        <translation>正在加载账户信息...</translation>
    </message>
    <message>
        <source>Manage account on terra-lab.ai</source>
        <translation>在 terra-lab.ai 上管理账户</translation>
    </message>
    <message>
        <source>Show</source>
        <translation>显示</translation>
    </message>
    <message>
        <source>Hide</source>
        <translation>隐藏</translation>
    </message>
    <message>
        <source>Change activation key</source>
        <translation>更改激活密钥</translation>
    </message>
    <message>
        <source>Plan</source>
        <translation>套餐</translation>
    </message>
    <message>
        <source>Free</source>
        <translation>免费</translation>
    </message>
    <message>
        <source>Canceled</source>
        <translation>已取消</translation>
    </message>
    <message>
        <source>Email</source>
        <translation>邮箱</translation>
    </message>
    <message>
        <source>Key</source>
        <translation>密钥</translation>
    </message>

    <!-- Corrupt checkpoint recovery (#65) -->
    <message>
        <source>Model File Corrupted</source>
        <translation>模型文件已损坏</translation>
    </message>
    <message>
        <source>Re-downloading Model</source>
        <translation>正在重新下载模型</translation>
    </message>
    <message>
        <source>The AI model file was corrupted and is being re-downloaded. Please try your selection again once it finishes.</source>
        <translation>AI 模型文件已损坏，正在重新下载。下载完成后请重新尝试选择。</translation>
    </message>
    <message>
        <source>The AI model file is corrupted but could not be removed automatically. Please delete this folder and restart QGIS:</source>
        <translation>AI 模型文件已损坏，但无法自动删除。请删除此文件夹并重启 QGIS：</translation>
    </message>

    <!-- One-click sign-in (browser pairing, mirrors AI Edit) -->
    <message>
        <source>Segment your map with AI</source>
        <translation>使用 AI 分割您的地图</translation>
    </message>
    <message>
        <source>Sign in / Sign up to start</source>
        <translation>登录 / 注册以开始</translation>
    </message>
    <message>
        <source>Sign in via your browser to start using AI Segmentation</source>
        <translation>通过浏览器登录以开始使用 AI Segmentation</translation>
    </message>
    <message>
        <source>Open again</source>
        <translation>再次打开</translation>
    </message>
    <message>
        <source>Didn't open? Open the page again</source>
        <translation>没有自动打开？重新打开页面</translation>
    </message>
    <message>
        <source>Couldn't open your browser. Use the manual key option below.</source>
        <translation>无法打开您的浏览器，请使用下方的手动输入密钥选项。</translation>
    </message>
    <message>
        <source>Sign-in timed out. Click Connect to try again.</source>
        <translation>登录超时，请点击“连接”重试。</translation>
    </message>
    <message>
        <source>Sign-in was cancelled in the browser. Click Connect to try again.</source>
        <translation>登录已在浏览器中取消，请点击“连接”重试。</translation>
    </message>
    <message>
        <source>Unexpected response from the server. Please try again.</source>
        <translation>服务器返回了意外的响应，请重试。</translation>
    </message>
    <message>
        <source>This account has no active AI Segmentation plan. Reactivate it on terra-lab.ai, then click Connect again.</source>
        <translation>该账户没有有效的 AI Segmentation 套餐，请在 terra-lab.ai 上重新激活，然后再次点击“连接”。</translation>
    </message>
    <message>
        <source>Connecting AI Segmentation</source>
        <translation>正在连接 AI Segmentation</translation>
    </message>
    <message>
        <source>Cancelling sign-in</source>
        <translation>正在取消登录</translation>
    </message>

    <!-- Help menu / account settings (mirrors AI Edit) -->
    <message>
        <source>Help / Report a problem</source>
        <translation>帮助 / 报告问题</translation>
    </message>
    <message>
        <source>Report a problem</source>
        <translation>报告问题</translation>
    </message>
    <message>
        <source>Connected</source>
        <translation>已连接</translation>
    </message>
    <message>
        <source>Sign out</source>
        <translation>退出登录</translation>
    </message>
    <message>
        <source>Sign out of AI Segmentation?</source>
        <translation>要退出 AI Segmentation 的登录吗？</translation>
    </message>
    <message>
        <source>You can sign back in anytime from QGIS.</source>
        <translation>您可以随时在 QGIS 中重新登录。</translation>
    </message>
    <message>
        <source>Active</source>
        <translation>有效</translation>
    </message>
    <message>
        <source>Free Trial</source>
        <translation>免费试用</translation>
    </message>
    <message>
        <source>Make this map presentation-ready</source>
        <translation>让这张地图达到展示级效果</translation>
    </message>
    <message>
        <source>AI Edit: turn your imagery into presentation and planning visuals</source>
        <translation>AI Edit：将您的影像转化为展示与规划用图</translation>
    </message>

    <!-- Pro / Automatic mode strings (plan #79) -->
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Mode selection</source>
        <translation>模式选择</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Choose between Manual (local) and Automatic (cloud) segmentation</source>
        <translation>在手动（本地）和自动（云端）分割之间选择</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Stop the active segmentation before switching modes.</source>
        <translation>请先停止当前的分割，再切换模式。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Cancel the active detection before switching modes.</source>
        <translation>请先取消当前的检测，再切换模式。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Sign in to use Automatic mode</source>
        <translation>登录以使用自动模式</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Your free detections are used up</source>
        <translation>您的免费检测次数已用完</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Subscribe to keep detecting without limits:</source>
        <translation>订阅以不受限制地持续检测：</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Subscribe to Pro</source>
        <translation>订阅 Pro</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Detect every building, tree, or road automatically</source>
        <translation>自动检测每一栋建筑、每一棵树或每一条道路</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>No installation required, no GPU, no limits</source>
        <translation>无需安装，无需 GPU，没有限制</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Built for large-scale digitization projects</source>
        <translation>专为大规模数字化项目而设计</translation>
    </message>
    <message>
        <location filename="../src/ui/zone_selection_maptool.py" line="0"/>
        <source>Clear this zone</source>
        <translation>清除该区域</translation>
    </message>
    <message>
        <location filename="../src/ui/zone_selection_maptool.py" line="0"/>
        <source>Cancel the running detection first</source>
        <translation>请先取消正在进行的检测</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>What do you want to detect?</source>
        <translation>您想检测什么？</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Where should the AI look?</source>
        <translation>AI 应该在哪里查找？</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Change</source>
        <translation>更改</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>What to detect...</source>
        <translation>要检测的对象...</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Hold the left mouse button and drag to draw a box on the map.</source>
        <translation>按住鼠标左键并拖动，在地图上绘制一个框。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>{n} tile(s) = {n} credit(s)</source>
        <translation>{n} 个瓦片 = {n} 积分</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Object: {obj}</source>
        <translation>对象：{obj}</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Detecting "{obj}"...</source>
        <translation>正在检测“{obj}”...</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Ground resolution per pixel. A smaller value lets the model detect smaller objects.</source>
        <translation>每像素对应的地面分辨率。数值越小，模型可检测的对象越小。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Detail</source>
        <translation>细节</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Higher detail splits the zone into more tiles. Each tile costs 1 credit and captures smaller objects.</source>
        <translation>更高的细节会将区域划分为更多瓦片。每个瓦片消耗 1 积分，并能捕捉更小的对象。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Zone too large - reduce the selection area</source>
        <translation>区域过大，请缩小选区范围</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Detect objects</source>
        <translation>检测对象</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Cancel detection</source>
        <translation>取消检测</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Tile {current}/{total}</source>
        <translation>瓦片 {current}/{total}</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Sending to the AI…</source>
        <translation>正在发送至 AI…</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>You're next · starting now…</source>
        <translation>轮到您了 · 正在开始…</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Spot reserved · starting in ~{eta}</source>
        <translation>名额已预留 · 约 {eta} 后开始</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Spot reserved · starting soon…</source>
        <translation>名额已预留 · 即将开始…</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>High demand · your spot is held…</source>
        <translation>需求高峰 · 您的名额已保留…</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Spot reserved · starting in a few seconds…</source>
        <translation>名额已预留 · 数秒后开始…</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>{s} seconds</source>
        <translation>{s} 秒</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>{m} min</source>
        <translation>{m} 分钟</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>{n} credits remaining (resets {date})</source>
        <translation>剩余 {n} 积分（{date} 重置）</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>{n} credits remaining</source>
        <translation>剩余 {n} 积分</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>{n} free detection(s) remaining</source>
        <translation>剩余 {n} 次免费检测</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Drawing...</source>
        <translation>正在绘制...</translation>
    </message>
    <message>
        <location filename="../src/ui/account_settings_dialog.py" line="0"/>
        <source>{n} free detection(s) remaining (lifetime)</source>
        <translation>剩余 {n} 次免费检测（永久额度）</translation>
    </message>
    <message>
        <location filename="../src/ui/account_settings_dialog.py" line="0"/>
        <source>Upgrade to Pro on terra-lab.ai</source>
        <translation>在 terra-lab.ai 上升级到 Pro</translation>
    </message>
    <message>
        <location filename="../src/ui/account_settings_dialog.py" line="0"/>
        <source>Pro</source>
        <translation>Pro</translation>
    </message>
    <message>
        <location filename="../src/ui/account_settings_dialog.py" line="0"/>
        <source>{remaining} / {total} credits</source>
        <translation>{remaining} / {total} 积分</translation>
    </message>
    <message>
        <location filename="../src/ui/account_settings_dialog.py" line="0"/>
        <source>resets {date}</source>
        <translation>{date} 重置</translation>
    </message>
    <message>
        <location filename="../src/ui/account_settings_dialog.py" line="0"/>
        <source>Free uses</source>
        <translation>免费使用次数</translation>
    </message>
    <message>
        <location filename="../src/ui/account_settings_dialog.py" line="0"/>
        <source>Credits</source>
        <translation>积分</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Auto detection (live)</source>
        <translation>自动检测（进行中）</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Preparing tiles...</source>
        <translation>正在准备瓦片...</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Could not render the zone. Try a smaller area or another layer.</source>
        <translation>无法渲染该区域，请尝试缩小范围或更换图层。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Cancelling...</source>
        <translation>正在取消...</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Finishing the previous run, please wait a moment...</source>
        <translation>正在结束上一次运行，请稍候...</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Found {n} object(s) but could not save the result file. Check folder permissions and the log.</source>
        <translation>发现 {n} 个对象，但无法保存结果文件。请检查文件夹权限和日志。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Could not save the result file. Check the log.</source>
        <translation>无法保存结果文件，请查看日志。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>No detections found. Try a different prompt or zoom level.</source>
        <translation>未发现任何检测结果，请尝试更换提示词或缩放级别。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Resume detection</source>
        <translation>恢复检测</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Detection failed. Check your connection and try again.</source>
        <translation>检测失败，请检查您的网络连接后重试。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Not enough credits to continue. The finished tiles are kept.</source>
        <translation>积分不足，无法继续。已完成的瓦片将被保留。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Zone too large. Reduce the area to 50 tiles or fewer.</source>
        <translation>区域过大，请将范围缩小到 50 个瓦片以内。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Automatic detection is temporarily unavailable. Please try again later.</source>
        <translation>自动检测暂时不可用，请稍后重试。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Draw a zone first. Automatic detection on online layers needs a zone.</source>
        <translation>请先绘制一个区域。在在线图层上进行自动检测需要指定区域。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>The zone is outside the selected raster layer. Pick the right layer or redraw the zone.</source>
        <translation>该区域位于所选栅格图层范围之外，请选择正确的图层或重新绘制区域。</translation>
    </message>
    <message>
        <source>Next</source>
        <translation>下一步</translation>
    </message>
    <message>
        <source>Export to layer</source>
        <translation>Export 到图层</translation>
    </message>
    <message>
        <source>{n} object(s) detected - adjust below, then export</source>
        <translation>检测到 {n} 个对象，在下方调整，然后 Export</translation>
    </message>
    <message>
        <source>Exported {n} polygon(s) to {name}</source>
        <translation>已将 {n} 个多边形 Export 到 {name}</translation>
    </message>
    <message>
        <source>Round corners</source>
        <translation>圆角</translation>
    </message>
    <message>
        <source>Fill holes</source>
        <translation>填充孔洞</translation>
    </message>
    <message>
        <source>Expand/Shrink:</source>
        <translation>扩展/收缩：</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Less</source>
        <translation>更少</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>More</source>
        <translation>更多</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>This area is large for this detail level. Raise detail or zoom in for sharper detections.</source>
        <translation>该区域相对于当前细节级别偏大。请提高细节或放大地图以获得更精确的检测结果。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>This zone is too large for sharp detections, even at maximum detail. Draw a smaller zone for the best results.</source>
        <translation>即使在最高细节级别下，该区域也过大，难以获得精确的检测结果。请绘制更小的区域以获得最佳效果。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Available when detection finishes</source>
        <translation>检测完成后可用</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Finishing up... {done}/{total}</source>
        <translation>正在收尾... {done}/{total}</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Min size:</source>
        <translation>最小尺寸：</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Max size:</source>
        <translation>最大尺寸：</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Off</source>
        <translation>关闭</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>No limit</source>
        <translation>无限制</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Hide detections smaller than this ground area. Use it to drop tiny noise blobs. 0 = keep all.</source>
        <translation>隐藏地面面积小于此值的检测结果，用于剔除微小的噪点。0 = 保留全部。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Hide detections larger than this ground area. 0 = no limit.</source>
        <translation>隐藏地面面积大于此值的检测结果。0 = 无限制。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Finer detail finds smaller objects and costs more credits.</source>
        <translation>更精细的细节可发现更小的对象，但会消耗更多积分。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>≈ {n} credits</source>
        <translation>≈ {n} 积分</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Finish</source>
        <translation>完成</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>{n} object(s) detected - adjust below, then Finish</source>
        <translation>检测到 {n} 个对象，在下方调整，然后点击“完成”</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Out of credits. Keep what was found below, then Finish.</source>
        <translation>积分已用完。保留下方已发现的结果，然后点击“完成”。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Saved {n} polygon(s) to {name}</source>
        <translation>已将 {n} 个多边形保存到 {name}</translation>
    </message>
    <message>
        <location filename="../src/api/terralab_client.py" line="0"/>
        <source>Cannot reach the server. Check your internet connection.</source>
        <translation>无法连接服务器，请检查您的网络连接。</translation>
    </message>
    <message>
        <location filename="../src/api/terralab_client.py" line="0"/>
        <source>Server refused the connection.</source>
        <translation>服务器拒绝了连接。</translation>
    </message>
    <message>
        <location filename="../src/api/terralab_client.py" line="0"/>
        <source>Request timed out. Check your connection or try again.</source>
        <translation>请求超时，请检查网络连接或重试。</translation>
    </message>
    <message>
        <location filename="../src/api/terralab_client.py" line="0"/>
        <source>SSL certificate error. Your network may be blocking secure connections.</source>
        <translation>SSL 证书错误，您的网络可能阻止了安全连接。</translation>
    </message>
    <message>
        <location filename="../src/api/terralab_client.py" line="0"/>
        <source>Proxy connection failed. Check QGIS proxy settings (Settings &gt; Options &gt; Network).</source>
        <translation>代理连接失败，请检查 QGIS 的代理设置（设置 &gt; 选项 &gt; 网络）。</translation>
    </message>
    <message>
        <location filename="../src/api/terralab_client.py" line="0"/>
        <source>Authentication failed. Please sign in again.</source>
        <translation>身份验证失败，请重新登录。</translation>
    </message>
    <message>
        <location filename="../src/api/terralab_client.py" line="0"/>
        <source>Network error. Check your internet connection.</source>
        <translation>网络错误，请检查您的网络连接。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Checking your AI Segmentation subscription</source>
        <translation>正在检查您的 AI Segmentation 订阅</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Loading AI Segmentation settings</source>
        <translation>正在加载 AI Segmentation 设置</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Refreshing credits</source>
        <translation>正在刷新积分</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Warming up AI Segmentation</source>
        <translation>正在预热 AI Segmentation</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>What do you want to segment?</source>
        <translation>您想分割什么？</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>e.g. building, solar panel (in English)</source>
        <translation>例如：building、solar panel（请使用英文）</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Popular</source>
        <translation>热门</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Library</source>
        <translation>对象库</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Browse objects with before / after examples.</source>
        <translation>浏览带有前后对比示例的对象。</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>The prompt is sent to the AI in English. Describe the object in 1-2 words (e.g. building, solar panel).</source>
        <translation>提示词会以英文发送给 AI。请用 1-2 个单词描述对象（例如 building、solar panel）。</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Use just 1-2 words for the object.</source>
        <translation>请仅用 1-2 个单词描述对象。</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Type the object itself, not a sentence or question.</source>
        <translation>请输入对象本身，而不是句子或问句。</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Too generic. Draw an example instead, or use a concrete word like building.</source>
        <translation>过于笼统。请改用绘制示例，或使用像 building 这样的具体词汇。</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Name a concrete object, not how it looks.</source>
        <translation>请命名一个具体对象，而不是描述它的外观。</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Segment one object - drop words like 'near' or 'with'.</source>
        <translation>一次分割一个对象，请去掉“near”或“with”之类的词。</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Use a real object word.</source>
        <translation>请使用真实存在的对象词汇。</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Use a 1-2 word object name.</source>
        <translation>请使用 1-2 个单词的对象名称。</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Did you mean '{term}'?</source>
        <translation>您是想输入“{term}”吗？</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Loading...</source>
        <translation>正在加载...</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>No preview</source>
        <translation>无预览</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>No preview yet</source>
        <translation>暂无预览</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Segment library</source>
        <translation>分割对象库</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>RECENT</source>
        <translation>最近</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Recently detected</source>
        <translation>最近检测</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>BROWSE</source>
        <translation>浏览</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Objects you detect will appear here.</source>
        <translation>您检测过的对象将显示在这里。</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>today</source>
        <translation>今天</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>yesterday</source>
        <translation>昨天</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>{n} days ago</source>
        <translation>{n} 天前</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>{n} detection(s)</source>
        <translation>{n} 个检测结果</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>{n} object(s)</source>
        <translation>{n} 个对象</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Loading segment library</source>
        <translation>正在加载分割对象库</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Pick an object to see a before / after, then use it.</source>
        <translation>选择一个对象查看前后对比，然后使用它。</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Search objects... e.g. building, solar panel</source>
        <translation>搜索对象... 例如 building、solar panel</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Prompt:</source>
        <translation>提示词：</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Fuzzy edges: this one may need cleanup after detection.</source>
        <translation>边缘模糊：此对象可能需要在检测后进行清理。</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Use this prompt</source>
        <translation>使用此提示词</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>No matching objects.</source>
        <translation>没有匹配的对象。</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Use</source>
        <translation>使用</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>≈ {n} tiles = {n} credits</source>
        <translation>≈ {n} 个瓦片 = {n} 积分</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Automatic mode scans your zone tile by tile. 1 tile = 1 credit, so this run costs about {n} credits. More detail splits the zone into more tiles, which costs more credits.</source>
        <translation>自动模式会逐个瓦片扫描您的区域。1 个瓦片 = 1 积分，因此本次运行大约消耗 {n} 积分。细节越高，区域被划分的瓦片越多，消耗的积分也越多。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Draw your example inside the selected zone.</source>
        <translation>在所选区域内绘制您的示例。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Clean edges:</source>
        <translation>清理边缘：</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Remove thin ragged fringe attached to the outline (0 = no change)</source>
        <translation>去除轮廓上细碎的毛边（0 = 不做更改）</translation>
    </message>
    <message>
        <location filename="../src/ui/account_settings_dialog.py" line="0"/>
        <source>Share anonymous usage statistics</source>
        <translation>分享匿名使用统计信息</translation>
    </message>
    <message>
        <location filename="../src/ui/account_settings_dialog.py" line="0"/>
        <source>Helps us fix bugs faster. Never includes your data, layers or coordinates.</source>
        <translation>帮助我们更快地修复问题。绝不包含您的数据、图层或坐标信息。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>{n} objects found</source>
        <translation>发现 {n} 个对象</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>No objects found</source>
        <translation>未发现任何对象</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>All shown at {pct}% confidence</source>
        <translation>全部以 {pct}% 置信度显示</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>{visible} shown at {pct}% · {hidden} more below this confidence</source>
        <translation>已显示 {visible} 个（置信度 {pct}%）· 另有 {hidden} 个低于该置信度</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>0 shown at {pct}% - lower Confidence to reveal them</source>
        <translation>在 {pct}% 下显示 0 个，降低置信度即可显示它们</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Started at {pct}% - nothing scored above.</source>
        <translation>起始置信度为 {pct}%，没有更高分的结果。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>More objects</source>
        <translation>更多对象</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Only confident</source>
        <translation>仅显示高置信度</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Objects not quite right? Refine them in Manual mode.</source>
        <translation>对象不够准确？在手动模式中细化它们。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Export {n} polygons</source>
        <translation>Export {n} 个多边形</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Lower Confidence to show objects first.</source>
        <translation>请先降低置信度以显示对象。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Adjust &amp; run again</source>
        <translation>调整并重新运行</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Discard these detections?</source>
        <translation>要放弃这些检测结果吗？</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Your {total} detections will be discarded. You keep your zone, object and settings. Running Detect again will use new credits.</source>
        <translation>您的 {total} 个检测结果将被放弃。您的区域、对象和设置会保留。再次运行检测将消耗新的积分。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Discard &amp;&amp; adjust</source>
        <translation>放弃并调整</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Keep your detections?</source>
        <translation>要保留检测结果吗？</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Save {visible} detections to a layer before leaving?</source>
        <translation>是否在离开前将 {visible} 个检测结果保存到图层？</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Save {total} detections (currently hidden by Confidence) to a layer before leaving?</source>
        <translation>是否在离开前将 {total} 个检测结果（当前被置信度筛选隐藏）保存到图层？</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Save &amp;&amp; exit</source>
        <translation>保存并退出</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Discard &amp;&amp; exit</source>
        <translation>放弃并退出</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>No {object} found in this zone</source>
        <translation>在此区域内未发现 {object}</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>This run used {n} credits. Things that usually fix it:</source>
        <translation>本次运行消耗了 {n} 积分。以下方法通常有效：</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Check the word is English and singular (building, not batiments)</source>
        <translation>检查该词是否为英文单数形式（例如 building，而非 batiments）</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Draw an example of one object (step 2)</source>
        <translation>绘制一个对象的示例（步骤 2）</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Raise Detail so small objects are visible</source>
        <translation>提高细节，使小对象可见</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Try a smaller or different zone</source>
        <translation>尝试更小或不同的区域</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Detecting "{obj}"... · {n} found so far</source>
        <translation>正在检测“{obj}”... · 目前已发现 {n} 个</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>How many objects sit at each confidence level.</source>
        <translation>各置信度级别下的对象数量分布。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>Your 500 free detections are used up</source>
        <translation>您的 500 次免费检测已用完</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>10,000 detections every month (~1,700 km2)</source>
        <translation>每月 10,000 次检测（约 1,700 平方公里）</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>Every building, tree, or road as clean polygons</source>
        <translation>将每一栋建筑、每一棵树或每一条道路转化为整洁的多边形</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>Cancel anytime; your exported layers stay yours</source>
        <translation>随时可取消；已 Export 的图层仍归您所有</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>Opens your TerraLab dashboard</source>
        <translation>打开您的 TerraLab 控制面板</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>Start Automatic AI Segmentation</source>
        <translation>开始自动 AI Segmentation</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>Draw your zone</source>
        <translation>绘制您的区域</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>Click on the map to outline the area to scan.</source>
        <translation>在地图上点击以圈出要扫描的区域。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>Keep clicking around the area, at least 3 points.</source>
        <translation>继续沿区域边缘点击，至少需要 3 个点。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>Click the first point to close the zone.</source>
        <translation>点击起始点以闭合区域。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>undo point</source>
        <translation>撤销点</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>cancel</source>
        <translation>取消</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/state.py" line="0"/>
        <source>Now outline one object, then click the first point to close.</source>
        <translation>现在圈出一个对象，然后点击起始点以闭合。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>Click on the map to drop points around the area you want to scan.</source>
        <translation>在地图上点击，沿要扫描的区域放置点。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>Exit</source>
        <translation>退出</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>e.g. building, tree, road, car</source>
        <translation>例如：building、tree、road、car</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>Browse ready-to-use objects with before / after previews.</source>
        <translation>浏览带有前后对比预览的现成对象。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>Draw an example</source>
        <translation>绘制示例</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>Subscribe to finish this zone: 10,000 credits/month.</source>
        <translation>订阅以完成该区域的检测：每月 10,000 积分。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>Blue = detected object</source>
        <translation>蓝色 = 已检测对象</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>Filter detections by confidence. Lower shows more (and noisier), higher keeps only the strongest. Free and instant.</source>
        <translation>按置信度筛选检测结果。数值越低，显示的结果越多（也更嘈杂）；数值越高，仅保留最可靠的结果。免费且即时生效。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>Show tiles (debug)</source>
        <translation>显示瓦片（调试）</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>Open these detections in Manual mode to fix specific objects point-by-point, then come back and export.</source>
        <translation>在手动模式中打开这些检测结果，逐点修正特定对象，然后返回进行 Export。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>Try instead:</source>
        <translation>请尝试：</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>"{word}" will run as "{token}".</source>
        <translation>“{word}” 将作为 “{token}” 运行。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>That word isn't recognized - try a common object like building or tree.</source>
        <translation>无法识别该词，请尝试常见对象，例如 building 或 tree。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>One object per run - start with the first one, then run again.</source>
        <translation>每次运行仅支持一个对象，请先处理第一个，然后再次运行。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>The Library has ready-to-use objects.</source>
        <translation>对象库中有现成的对象可供使用。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>Keep clicking to add points ({n} so far, 3 minimum).</source>
        <translation>继续点击以添加点（目前 {n} 个，至少需要 3 个）。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>{n} points. Double-click or press Enter to finish, or click the first point to close.</source>
        <translation>已添加 {n} 个点。双击或按 Enter 键完成，或点击起始点以闭合。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>~ {credits} credits</source>
        <translation>~ {credits} 积分</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>{remaining} left</source>
        <translation>剩余 {remaining}</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>{remaining} free left</source>
        <translation>剩余 {remaining} 次免费</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>~ {credits} credits · </source>
        <translation>~ {credits} 积分 · </translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>will stop after {remaining}</source>
        <translation>将在 {remaining} 后停止</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>1 credit ~ 0.17 km2 at default detail.</source>
        <translation>默认细节级别下，1 积分约对应 0.17 平方公里。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>Tip: draw an example of one {object} to catch more of them.</source>
        <translation>提示：绘制一个 {object} 的示例，以发现更多同类对象。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>object</source>
        <translation>对象</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>Dense area: raise Detail to catch every object.</source>
        <translation>密集区域：提高细节以捕捉每一个对象。</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/cards.py" line="0"/>
        <source>Restore</source>
        <translation>恢复</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/cards.py" line="0"/>
        <source>Delete forever</source>
        <translation>永久删除</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/cards.py" line="0"/>
        <source>Deleted {when} · purges in {n} days</source>
        <translation>已于 {when} 删除 · {n} 天后彻底清除</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/cards.py" line="0"/>
        <source>{tiles} tiles · {objects} objects · {credits} credits</source>
        <translation>{tiles} 个瓦片 · {objects} 个对象 · {credits} 积分</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/cards.py" line="0"/>
        <source>Older detection</source>
        <translation>较早的检测记录</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Details</source>
        <translation>详情</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Fullscreen</source>
        <translation>全屏</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Exit fullscreen</source>
        <translation>退出全屏</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Prompt</source>
        <translation>提示词</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Copy</source>
        <translation>复制</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Copy prompt</source>
        <translation>复制提示词</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Copied</source>
        <translation>已复制</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Template</source>
        <translation>模板</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Your detection</source>
        <translation>您的检测结果</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Open the Library from the Automatic page to use this.</source>
        <translation>从自动模式页面打开对象库以使用此项。</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>DATE</source>
        <translation>日期</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>OBJECTS</source>
        <translation>对象</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>CREDITS</source>
        <translation>积分</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>TILES</source>
        <translation>瓦片</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>RESOLUTION</source>
        <translation>分辨率</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>EXAMPLE</source>
        <translation>示例</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Used</source>
        <translation>已使用</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Restore to map</source>
        <translation>恢复到地图</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Reopens this run's review at the same place. Free - no credits.</source>
        <translation>在原位置重新打开本次运行的复核界面。免费，不消耗积分。</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Export...</source>
        <translation>Export...</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Delete</source>
        <translation>删除</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Drop this object back into the prompt box for a new detection.</source>
        <translation>将此对象重新放入提示词框，以进行新的检测。</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Remove from favorites</source>
        <translation>从收藏中移除</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Add to favorites</source>
        <translation>添加到收藏</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Format:</source>
        <translation>格式：</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>GeoPackage keeps the embedded style; other formats are saved without a style.</source>
        <translation>GeoPackage 会保留内嵌样式；其他格式保存时不带样式。</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Browse...</source>
        <translation>浏览...</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/dialog.py" line="0"/>
        <source>Your detections</source>
        <translation>您的检测结果</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/dialog.py" line="0"/>
        <source>Recent</source>
        <translation>最近</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/dialog.py" line="0"/>
        <source>Favorites</source>
        <translation>收藏</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/dialog.py" line="0"/>
        <source>Recently deleted</source>
        <translation>最近删除</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/dialog.py" line="0"/>
        <source>Templates</source>
        <translation>模板</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/dialog.py" line="0"/>
        <source>Load older runs</source>
        <translation>加载更早的运行记录</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/dialog.py" line="0"/>
        <source>Nothing here yet. Your automatic detections will land here, ready to reuse, restore or export.</source>
        <translation>这里还没有内容。您的自动检测结果会显示在这里，可随时重用、恢复或 Export。</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/dialog.py" line="0"/>
        <source>Star a detection to keep it here.</source>
        <translation>将检测结果加星标以保留在这里。</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/dialog.py" line="0"/>
        <source>Deleted runs wait here for 30 days, then they are purged for good.</source>
        <translation>已删除的运行记录会在这里保留 30 天，之后将被永久清除。</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/dialog.py" line="0"/>
        <source>This permanently removes the stored previews and masks. Exported layers are never touched.</source>
        <translation>这将永久删除已存储的预览图和掩膜。已 Export 的图层不会受到影响。</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/dialog.py" line="0"/>
        <source>Could not load this run's stored detections. Try again later.</source>
        <translation>无法加载此次运行存储的检测结果，请稍后重试。</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/dialog.py" line="0"/>
        <source>Nothing to export at this confidence. Lower it and try again.</source>
        <translation>当前置信度下没有可 Export 的内容，请降低置信度后重试。</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/dialog.py" line="0"/>
        <source>The export failed. Check the file path and try again.</source>
        <translation>Export 失败，请检查文件路径后重试。</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/dialog.py" line="0"/>
        <source>Exported {n} polygon(s).</source>
        <translation>已 Export {n} 个多边形。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>Add a point</source>
        <translation>添加一个点</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>Add area</source>
        <translation>增加面积</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/state.py" line="0"/>
        <source>All detections kept. Go 'Back to review' to export.</source>
        <translation>所有检测结果均已保留。点击“返回复核”以 Export。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>Arrow keys</source>
        <translation>方向键</translation>
    </message>
    <message>
        <location filename="../src/api/terralab_client.py" line="0"/>
        <source>Automatic</source>
        <translation>自动</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>Automatic - detect and review</source>
        <translation>自动 - 检测与复核</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>Automatic - draw your zone</source>
        <translation>自动 - 绘制您的区域</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>Cancel the drawing</source>
        <translation>取消绘制</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>Cancel the running detection, or exit the review</source>
        <translation>取消正在进行的检测，或退出复核</translation>
    </message>
    <message>
        <location filename="../src/core/feature_encoder.py" line="0"/>
        <source>Click</source>
        <translation>点击</translation>
    </message>
    <message>
        <location filename="../src/ui/plugin/auto_run.py" line="0"/>
        <source>Could not place the example on the image. Redraw the example box inside the zone and try again.</source>
        <translation>无法将示例放置到图像上，请在区域内重新绘制示例框并重试。</translation>
    </message>
    <message>
        <location filename="../src/core/run_restore.py" line="0"/>
        <source>Could not rebuild this run's detections.</source>
        <translation>无法重建此次运行的检测结果。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/state.py" line="0"/>
        <source>Delete removes this object</source>
        <translation>删除键会移除该对象</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>Delete the active object</source>
        <translation>删除当前选中的对象</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>Detect objects, or export the reviewed polygons</source>
        <translation>检测对象，或 Export 已复核的多边形</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Detection continues in the background. Reopen AI Segmentation to follow it.</source>
        <translation>检测将在后台继续进行。重新打开 AI Segmentation 以查看进度。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>Double-click</source>
        <translation>双击</translation>
    </message>
    <message>
        <location filename="../src/core/pip_diagnostics.py" line="0"/>
        <source>Example</source>
        <translation>示例</translation>
    </message>
    <message>
        <location filename="../src/core/run_restore.py" line="0"/>
        <source>Finish or exit the current run before restoring a past one.</source>
        <translation>请先完成或退出当前运行，再恢复以往的运行记录。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>Finish the zone</source>
        <translation>完成该区域</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>General</source>
        <translation>常规</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/build.py" line="0"/>
        <source>Hand-refined objects are always kept, whatever the confidence.</source>
        <translation>手动细化过的对象将始终被保留，无论置信度高低。</translation>
    </message>
    <message>
        <location filename="../src/ui/plugin/manual_handoff.py" line="0"/>
        <source>Install now</source>
        <translation>立即安装</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/build.py" line="0"/>
        <source>Keeps this polygon in your session. Export writes all kept polygons to a layer.</source>
        <translation>将此多边形保留在本次会话中。Export 会将所有保留的多边形写入图层。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_maptool.py" line="0"/>
        <source>Keyboard shortcuts</source>
        <translation>键盘快捷键</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>Left-click</source>
        <translation>左键点击</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/state.py" line="0"/>
        <source>Left-click adds area · Right-click removes area</source>
        <translation>左键点击增加面积 · 右键点击减少面积</translation>
    </message>
    <message>
        <location filename="../src/core/layer_conventions.py" line="0"/>
        <source>Manual</source>
        <translation>手动</translation>
    </message>
    <message>
        <location filename="../src/ui/plugin/manual_handoff.py" line="0"/>
        <source>Manual mode needs a one-time setup</source>
        <translation>手动模式需要进行一次性安装</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Manual session</source>
        <translation>手动会话</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>Max detail for this zone - draw a larger zone for finer detail.</source>
        <translation>该区域已达到最高细节级别，绘制更大的区域可获得更精细的细节。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/build.py" line="0"/>
        <source>Merges your edits back into the review. Nothing is exported yet.</source>
        <translation>将您的编辑合并回复核结果，目前尚未 Export。</translation>
    </message>
    <message>
        <location filename="../src/core/venv_manager.py" line="0"/>
        <source>OK</source>
        <translation>确定</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_review_build.py" line="0"/>
        <source>One color per object - check neighbors are separated</source>
        <translation>每个对象使用一种颜色，检查相邻对象是否已分开</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_review_build.py" line="0"/>
        <source>Optional shape and size controls: simplify outlines, clean edges, round corners, expand or shrink, fill holes, size filters.</source>
        <translation>可选的形状与尺寸控制：简化轮廓、清理边缘、圆角处理、扩展或收缩、填充孔洞、尺寸筛选。</translation>
    </message>
    <message>
        <location filename="../src/ui/plugin/auto_lifecycle.py" line="0"/>
        <source>Out of credits after {done}/{total} tiles. Your detections are kept below.</source>
        <translation>处理完 {done}/{total} 个瓦片后积分已用完，已发现的检测结果保留在下方。</translation>
    </message>
    <message>
        <location filename="../src/ui/plugin/auto_results.py" line="0"/>
        <source>Outlines only - check boundaries against the imagery</source>
        <translation>仅显示轮廓，对照影像检查边界</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>Pan the map</source>
        <translation>平移地图</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Part of your zone is outside "{layer}" - only the overlapping area will return objects.</source>
        <translation>您的区域部分超出“{layer}”范围，仅重叠部分会返回检测结果。</translation>
    </message>
    <message>
        <location filename="../src/ui/plugin/auto_run.py" line="0"/>
        <source>Pick an object to detect first (nothing was selected).</source>
        <translation>请先选择要检测的对象（当前未选择任何内容）。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/state.py" line="0"/>
        <source>Polygon saved ({n} total). Click another element, or export when done.</source>
        <translation>多边形已保存（共 {n} 个）。点击其他对象继续，或完成后 Export。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/state.py" line="0"/>
        <source>Press S to keep it (turns green)</source>
        <translation>按 S 键保留（变为绿色）</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_review_build.py" line="0"/>
        <source>Refine detections</source>
        <translation>细化检测结果</translation>
    </message>
    <message>
        <location filename="../src/ui/plugin/auto_results.py" line="0"/>
        <source>Refine seeds</source>
        <translation>细化种子点</translation>
    </message>
    <message>
        <location filename="../src/ui/plugin/manual_handoff.py" line="0"/>
        <source>Refining uses the free local AI, which is not installed yet. Install it now (a few minutes, automatic)? Your detections stay safely in the review.</source>
        <translation>细化功能使用免费的本地 AI，目前尚未安装。是否现在安装（自动完成，约需几分钟）？您的检测结果会安全保留在复核界面中。</translation>
    </message>
    <message>
        <location filename="../src/core/checkpoint_manager.py" line="0"/>
        <source>Remove</source>
        <translation>移除</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>Remove area</source>
        <translation>移除面积</translation>
    </message>
    <message>
        <location filename="../src/ui/zone_selection_maptool.py" line="0"/>
        <source>Remove zone</source>
        <translation>移除区域</translation>
    </message>
    <message>
        <location filename="../src/ui/account_settings_dialog.py" line="0"/>
        <source>Resets {date}</source>
        <translation>{date} 重置</translation>
    </message>
    <message>
        <location filename="../src/core/run_restore.py" line="0"/>
        <source>Restored "{prompt}" - adjust and export below.</source>
        <translation>已恢复“{prompt}”，请在下方调整并 Export。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>Right-click</source>
        <translation>右键点击</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/build.py" line="0"/>
        <source>Save polygon (S)</source>
        <translation>保存多边形（S）</translation>
    </message>
    <message>
        <location filename="../src/ui/plugin/auto_lifecycle.py" line="0"/>
        <source>Session expired. Sign in again to continue.</source>
        <translation>会话已过期，请重新登录以继续。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>Start (the visible mode's Start button)</source>
        <translation>开始（当前可见模式的“开始”按钮）</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/build.py" line="0"/>
        <source>Start Manual AI Segmentation</source>
        <translation>开始手动 AI Segmentation</translation>
    </message>
    <message>
        <location filename="../src/ui/plugin/auto_lifecycle.py" line="0"/>
        <source>The raster was removed. Your polygons were saved to a layer.</source>
        <translation>该栅格已被移除，您的多边形已保存到图层中。</translation>
    </message>
    <message>
        <location filename="../src/ui/plugin/auto_lifecycle.py" line="0"/>
        <source>The selected raster was removed.</source>
        <translation>所选栅格已被移除。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>The selected raster was removed. Keeping what was already found.</source>
        <translation>所选栅格已被移除，已发现的结果将被保留。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/build.py" line="0"/>
        <source>Tip: S saves, Enter exports, Ctrl+Z undoes a click.</source>
        <translation>提示：S 键保存，Enter 键 Export，Ctrl+Z 撤销上一次点击。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>Undo the last point</source>
        <translation>撤销上一个点</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/state.py" line="0"/>
        <source>Writes a GeoPackage layer with your {n} kept polygons.</source>
        <translation>将您保留的 {n} 个多边形写入一个 GeoPackage 图层。</translation>
    </message>
    <message>
        <location filename="../src/ui/plugin/auto_results.py" line="0"/>
        <source>Yellow = confident · Purple = uncertain</source>
        <translation>黄色 = 可信 · 紫色 = 不确定</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Your zone is outside "{layer}". Pick the right layer or draw inside it.</source>
        <translation>您的区域位于“{layer}”范围之外，请选择正确的图层，或在其范围内重新绘制。</translation>
    </message>
    <message>
        <location filename="../src/ui/plugin/auto_zone.py" line="0"/>
        <source>Zone too large. Reduce the area to {max} tiles or fewer.</source>
        <translation>区域过大，请将范围缩小到 {max} 个瓦片以内。</translation>
    </message>
    <message>
        <location filename="../src/api/terralab_client.py" line="0"/>
        <source>objects</source>
        <translation>对象</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>{kept} of {total} detections kept - click a blue detection to edit it</source>
        <translation>已保留 {kept}/{total} 个检测结果，点击蓝色检测结果以编辑</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>{n} of {total} free detections left</source>
        <translation>剩余 {n}/{total} 次免费检测</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>{remaining} free detections left.</source>
        <translation>剩余 {remaining} 次免费检测。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>≈ 1 tile = 1 credit</source>
        <translation>≈ 1 个瓦片 = 1 积分</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_review_build.py" line="0"/>
        <source>Keep this result</source>
        <translation>保留此结果</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_review_build.py" line="0"/>
        <source>Start over</source>
        <translation>重新开始</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_review_build.py" line="0"/>
        <source>Adjust and run again</source>
        <translation>调整并重新运行</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>all shown</source>
        <translation>全部已显示</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>{visible} of {n} shown</source>
        <translation>已显示 {visible}/{n} 个</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>{hidden} below {pct}%</source>
        <translation>{hidden} 个低于 {pct}%</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_review_build.py" line="0"/>
        <source>Adjust and run again</source>
        <translation>调整并重新运行</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_review_build.py" line="0"/>
        <source>Keep this result</source>
        <translation>保留此结果</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_review_build.py" line="0"/>
        <source>Start over</source>
        <translation>重新开始</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_review_build.py" line="0"/>
        <source>all shown</source>
        <translation>全部已显示</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_review_build.py" line="0"/>
        <source>{hidden} below {pct}%</source>
        <translation>{hidden} 个低于 {pct}%</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_review_build.py" line="0"/>
        <source>{visible} of {n} shown</source>
        <translation>已显示 {visible}/{n} 个</translation>
    </message>
    <message>
        <source>Right angles:</source>
        <translation>直角化：</translation>
    </message>
    <message>
        <source>Snap edges to 90 degrees for man-made shapes like buildings, pools and solar panels.</source>
        <translation>将边缘吸附为 90 度直角，适用于建筑、泳池、太阳能板等人工形状。</translation>
    </message>
    <message>
        <source>Click an object on the map and the AI outlines it. You go one object at a time, checking and saving each polygon yourself.</source>
        <translation>在地图上点击一个对象，AI 会为其生成轮廓。您可以逐个处理对象，亲自检查并保存每个多边形。</translation>
    </message>
    <!-- Onboarding conversion batch (2026-07-04) -->
    <message>
        <source>Show guidance tips again</source>
        <translation>再次显示引导提示</translation>
    </message>
    <message>
        <source>Guidance tips restored</source>
        <translation>引导提示已恢复</translation>
    </message>
    <message>
        <source>Run again here</source>
        <translation>在此处再次运行</translation>
    </message>
    <message>
        <source>Reload this zone and object, ready to detect.</source>
        <translation>重新加载该区域和对象，可立即开始检测。</translation>
    </message>
    <message>
        <source>Same object, new zone</source>
        <translation>相同对象，新区域</translation>
    </message>
    <message>
        <source>Keep this object and draw a new zone on the map.</source>
        <translation>保留此对象，并在地图上绘制新的区域。</translation>
    </message>
    <message>
        <source>Upgrade to Pro</source>
        <translation>升级到 Pro</translation>
    </message>
    <message>
        <source>Free account - sign up takes 15 seconds in your browser.</source>
        <translation>免费账户，在浏览器中注册仅需 15 秒。</translation>
    </message>
    <message>
        <source>Manual mode stays free and unlimited on your computer.</source>
        <translation>手动模式在您的电脑上始终免费且不受限制。</translation>
    </message>
    <message>
        <source>Finds every object of one kind in your zone - draw a zone, name the object, get all the polygons at once.</source>
        <translation>一次性找出您区域内某一类别的所有对象，绘制区域、命名对象，即可一次获得全部多边形。</translation>
    </message>
    <message>
        <source>Tip: draw an example on the map to boost detection of unusual objects.</source>
        <translation>提示：在地图上绘制一个示例，可提升对特殊对象的检测效果。</translation>
    </message>
    <message>
        <source>Tip: lower Confidence to reveal more detections, raise it to keep only the best.</source>
        <translation>提示：降低置信度可显示更多检测结果，提高置信度则只保留最佳结果。</translation>
    </message>
    <message>
        <source>This zone is {area} km2 - free trial zones go up to {max} km2.</source>
        <translation>该区域面积为 {area} 平方公里，免费试用区域上限为 {max} 平方公里。</translation>
    </message>
    <message>
        <source>Draw a smaller zone, or &lt;a href=&quot;{url}&quot;&gt;subscribe&lt;/a&gt; to segment areas of any size.</source>
        <translation>请绘制更小的区域，或&lt;a href=&quot;{url}&quot;&gt;订阅&lt;/a&gt;以分割任意大小的区域。</translation>
    </message>
    <message>
        <source>Running low: {n} free detections left. &lt;a href=&quot;{url}&quot;&gt;Subscribe&lt;/a&gt; to keep going.</source>
        <translation>免费次数即将用完：剩余 {n} 次免费检测。&lt;a href=&quot;{url}&quot;&gt;订阅&lt;/a&gt;以继续使用。</translation>
    </message>
    <message>
        <source>Last run: {count} {object} exported · {area} km2 · {used} credits used, {left} left</source>
        <translation>上次运行：已 Export {count} 个 {object} · {area} 平方公里 · 消耗 {used} 积分，剩余 {left}</translation>
    </message>
    <message>
        <source>Last run: {count} {object} exported · {area} km2 · {used} credits used</source>
        <translation>上次运行：已 Export {count} 个 {object} · {area} 平方公里 · 消耗 {used} 积分</translation>
    </message>
    <message>
        <source>Then segment any imagery: point and click, or fully automatic.</source>
        <translation>然后即可分割任意影像：点选操作，或全自动完成。</translation>
    </message>
    <message>
        <source>Waiting for your browser sign-in...</source>
        <translation>正在等待您在浏览器中完成登录...</translation>
    </message>
    <message>
        <source>New: Automatic mode finds every object in a zone at once.</source>
        <translation>新功能：自动模式可一次性找出区域内的所有对象。</translation>
    </message>
    <message>
        <source>Try Automatic</source>
        <translation>试用自动模式</translation>
    </message>
    <message>
        <source>Got it - hide this tip</source>
        <translation>知道了，隐藏此提示</translation>
    </message>
    <message>
        <source>Finish or cancel the current detection before re-running a past one.</source>
        <translation>请先完成或取消当前检测，再重新运行以往的记录。</translation>
    </message>
    <message>
        <source>Couldn&apos;t open your browser. Check your connection and click Sign in / Sign up to start again.</source>
        <translation>无法打开您的浏览器，请检查网络连接，然后点击“登录 / 注册”重新开始。</translation>
    </message>
</context>
</TS>
