<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE TS>
<TS version="2.1" language="zh_CN">
<context>
    <name>AISegmentation</name>
    <!-- Review display colours: Normal / Confidence / Random (2026-07-01) -->
    <message>
        <source>Normal</source>
        <translation>普通</translation>
    </message>
    <message>
        <source>Confidence</source>
        <translation>置信度</translation>
    </message>
    <message>
        <source>Outline</source>
        <translation>轮廓</translation>
    </message>
    <!-- Retry: back to setup keeping inputs (2026-07-01) -->
    <message>
        <source>Retry</source>
        <translation>重试</translation>
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
        <source>Exclude a look-alike</source>
        <translation>排除相似对象</translation>
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
        <source>Mark an object to find more like it.</source>
        <translation>标记一个对象，以查找更多相似对象。</translation>
    </message>
    <message>
        <source>Mark a false positive to drop things like it.</source>
        <translation>标记一个误检结果，以剔除类似的结果。</translation>
    </message>
    <message>
        <source>Finer detail finds smaller objects.</source>
        <translation>更精细的细节可发现更小的对象。</translation>
    </message>
    <!-- Refine in Manual handoff -->
    <message>
        <source>Finish or exit the review to switch modes.</source>
        <translation>请先完成或退出复核，再切换模式。</translation>
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
        <source>Select a raster layer (GeoTIFF, WMS, XYZ tiles, etc.)</source>
        <translation>选择一个栅格图层（GeoTIFF、WMS、XYZ 瓦片等）</translation>
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
        <source>Space</source>
        <translation>空格键</translation>
    </message>
    <message>
        <source>Hold and move to pan the map</source>
        <translation>按住并移动以平移地图</translation>
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
        <source>Fill interior holes in the selection</source>
        <translation>填充选区内部的孔洞</translation>
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
        <source>Outline</source>
        <translation>轮廓</translation>
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
        <source>Terms</source>
        <translation>条款</translation>
    </message>
    <message>
        <source>Privacy</source>
        <translation>隐私</translation>
    </message>

    <!-- Tooltip -->
    <message>
        <source>Segment elements on raster images using AI</source>
        <translation>使用 AI 分割栅格图像中的对象</translation>
    </message>

    <!-- Error Report Dialog -->
    <message>
        <source>Copy your logs with the button below and send them to our support email.</source>
        <translation>使用下方按钮复制您的日志,并发送到我们的支持邮箱。</translation>
    </message>
    <message>
        <source>We'll get this fixed for you :)</source>
        <translation>我们会为您修复这个问题 :)</translation>
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
        <source>Missing Visual C++ Redistributable. Install it, restart your computer, then click Retry.</source>
        <translation>缺少 Visual C++ Redistributable。请安装后重启电脑，然后点击“重试”。</translation>
    </message>

    <message>
        <source>Retry</source>
        <translation>重试</translation>
    </message>
<!-- v1.0.0 strings -->
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
        <source>Checking...</source>
        <translation>正在检查...</translation>
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
        <source>Subscribe to keep detecting without limits:</source>
        <translation>订阅以不受限制地持续检测：</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Subscribe to Pro</source>
        <translation>订阅 Pro</translation>
    </message>
    <message>
        <location filename="../src/ui/zone_selection_maptool.py" line="0"/>
        <source>Cancel the running detection first</source>
        <translation>请先取消正在进行的检测</translation>
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
        <source>Spot reserved · starting in ~{eta}</source>
        <translation>名额已预留 · 约 {eta} 后开始</translation>
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
        <source>{n} credits remaining</source>
        <translation>剩余 {n} 积分</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>{n} free detection(s) remaining</source>
        <translation>剩余 {n} 次免费检测</translation>
    </message>
    <message>
        <location filename="../src/ui/account_settings_dialog.py" line="0"/>
        <source>{remaining} / {total} credits</source>
        <translation>{remaining} / {total} 积分</translation>
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
        <source>Detection failed. Check your connection and try again.</source>
        <translation>检测失败，请检查您的网络连接后重试。</translation>
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
        <source>Search objects... e.g. building, solar panel</source>
        <translation>搜索对象... 例如 building、solar panel</translation>
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
        <source>How many objects sit at each confidence level.</source>
        <translation>各置信度级别下的对象数量分布。</translation>
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
        <source>Subscribe to finish this zone: 10,000 credits/month.</source>
        <translation>订阅以完成该区域的检测：每月 10,000 积分。</translation>
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
        <source>object</source>
        <translation>对象</translation>
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
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>Delete the active object</source>
        <translation>删除当前选中的对象</translation>
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
        <location filename="../src/core/layer_conventions.py" line="0"/>
        <source>Manual</source>
        <translation>手动</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>Max detail for this zone - draw a larger zone for finer detail.</source>
        <translation>该区域已达到最高细节级别，绘制更大的区域可获得更精细的细节。</translation>
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
        <location filename="../src/ui/plugin/auto_results.py" line="0"/>
        <source>Refine seeds</source>
        <translation>细化种子点</translation>
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
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>{n} of {total} free detections left</source>
        <translation>剩余 {n}/{total} 次免费检测</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>≈ 1 tile = 1 credit</source>
        <translation>≈ 1 个瓦片 = 1 积分</translation>
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
        <location filename="../src/ui/dock/auto_review_build.py" line="0"/>
        <source>all shown</source>
        <translation>全部已显示</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_review_build.py" line="0"/>
        <source>{visible} of {n} shown</source>
        <translation>已显示 {visible}/{n} 个</translation>
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
        <source>Draw a smaller zone, or &lt;a href=&quot;{url}&quot;&gt;subscribe&lt;/a&gt; to segment areas of any size.</source>
        <translation>请绘制更小的区域，或&lt;a href=&quot;{url}&quot;&gt;订阅&lt;/a&gt;以分割任意大小的区域。</translation>
    </message>
    <message>
        <source>Running low: {n} free detections left. &lt;a href=&quot;{url}&quot;&gt;Subscribe&lt;/a&gt; to keep going.</source>
        <translation>免费次数即将用完：剩余 {n} 次免费检测。&lt;a href=&quot;{url}&quot;&gt;订阅&lt;/a&gt;以继续使用。</translation>
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
        <source>0 shown - lower the Min size filter to reveal them</source>
        <translation>显示 0 个，降低最小尺寸筛选即可显示它们</translation>
    </message>
    <message>
        <source>10,000 credits every month. Cancel anytime.</source>
        <translation>每月 10,000 积分，随时可取消。</translation>
    </message>
    <message>
        <source>A Component Failed to Load</source>
        <translation>组件加载失败</translation>
    </message>
    <message>
        <source>AI Segmentation</source>
        <translation>AI Segmentation</translation>
    </message>
    <message>
        <source>AI data removed, but some items could not be fully cleared. You can delete the folder manually.</source>
        <translation>AI 数据已移除，但部分项目未能完全清除。您可以手动删除该文件夹。</translation>
    </message>
    <message>
        <source>Accept the Terms and Privacy Policy first.</source>
        <translation>请先接受服务条款和隐私政策。</translation>
    </message>
    <message>
        <source>An install or detection is still running. Wait for it to finish, then try again.</source>
        <translation>安装或检测仍在进行中，请等待其完成后再重试。</translation>
    </message>
    <message>
        <source>Any GeoTIFF, WMS or XYZ basemap.</source>
        <translation>支持任意 GeoTIFF、WMS 或 XYZ 底图。</translation>
    </message>
    <message>
        <source>At this detail {obj} is too small to spot - raise the detail.</source>
        <translation>在当前细节级别下，{obj}太小而难以发现，请提高细节。</translation>
    </message>
    <message>
        <source>Available once the current install or detection finishes.</source>
        <translation>当前的安装或检测完成后即可使用。</translation>
    </message>
    <message>
        <source>Blocked by IT Security Policy</source>
        <translation>已被 IT 安全策略阻止</translation>
    </message>
    <message>
        <source>Browse the library (view only while detecting).</source>
        <translation>浏览对象库（检测过程中仅可查看）。</translation>
    </message>
    <message>
        <source>Could not reach the service. Check your connection and try again.</source>
        <translation>无法连接到服务，请检查您的网络连接后重试。</translation>
    </message>
    <message>
        <source>Could not read pixels from this {ext} file. The file may be corrupt, truncated, or use a compression your GDAL build cannot decode.
Try opening it in QGIS to confirm it displays, or convert it to GeoTIFF (.tif) before using AI Segmentation.</source>
        <translation>无法从此 {ext} 文件读取像素。文件可能已损坏、被截断，或使用了您的 GDAL 版本无法解码的压缩方式。
请尝试在 QGIS 中打开以确认能否正常显示，或在使用 AI Segmentation 之前将其转换为 GeoTIFF（.tif）。</translation>
    </message>
    <message>
        <source>Could not remove the AI data. Try again.</source>
        <translation>无法移除 AI 数据，请重试。</translation>
    </message>
    <message>
        <source>Couldn't load the demo imagery. Check your internet connection, or add your own layer.</source>
        <translation>无法加载演示影像，请检查您的网络连接，或添加您自己的图层。</translation>
    </message>
    <message>
        <source>Detected object</source>
        <translation>已检测对象</translation>
    </message>
    <message>
        <source>Detection failed. Please try again.</source>
        <translation>检测失败，请重试。</translation>
    </message>
    <message>
        <source>Downloaded AI data removed. You have been signed out.</source>
        <translation>已移除下载的 AI 数据，您已退出登录。</translation>
    </message>
    <message>
        <source>Draw on the map</source>
        <translation>在地图上绘制</translation>
    </message>
    <message>
        <source>Extra detail keeps helping {obj} in this zone.</source>
        <translation>在此区域中，提高细节仍能持续改善{obj}的检测效果。</translation>
    </message>
    <message>
        <source>Free plan</source>
        <translation>免费套餐</translation>
    </message>
    <message>
        <source>Hide parts larger than this ground area. 0 = no limit.</source>
        <translation>隐藏地面面积大于此值的部分。0 = 无限制。</translation>
    </message>
    <message>
        <source>Hide parts smaller than this ground area. Use it to drop tiny noise blobs. 0 = keep all.</source>
        <translation>隐藏地面面积小于此值的部分，用于剔除微小的噪点。0 = 保留全部。</translation>
    </message>
    <message>
        <source>Load example imagery</source>
        <translation>加载示例影像</translation>
    </message>
    <message>
        <source>Load your own imagery</source>
        <translation>加载您自己的影像</translation>
    </message>
    <message>
        <source>Lower the Min size filter to show objects first.</source>
        <translation>请先降低最小尺寸筛选以显示对象。</translation>
    </message>
    <message>
        <source>Manage account</source>
        <translation>管理账户</translation>
    </message>
    <message>
        <source>Manual Mode Not Supported</source>
        <translation>不支持手动模式</translation>
    </message>
    <message>
        <source>Manual mode installs local components that are not available for this Mac with this version of QGIS. Please use Automatic mode instead, which runs fully in the cloud and needs no local install.</source>
        <translation>手动模式需要安装本地组件，但在此 Mac 与此版本的 QGIS 下不可用。请改用自动模式，它完全在云端运行，无需本地安装。</translation>
    </message>
    <message>
        <source>Manual mode is not supported in this QGIS installation</source>
        <translation>此 QGIS 安装环境不支持手动模式</translation>
    </message>
    <message>
        <source>Manual mode needs to install local dependencies, which is not supported inside this sandboxed QGIS installation (Flatpak or Snap). Please use Automatic mode instead, which runs fully in the cloud and needs no local install.</source>
        <translation>手动模式需要安装本地依赖项，但在此沙盒化的 QGIS 环境（Flatpak 或 Snap）中不受支持。请改用自动模式，它完全在云端运行，无需本地安装。</translation>
    </message>
    <message>
        <source>Name the object (or draw an example) first - Detail then tunes itself to it.</source>
        <translation>请先命名对象（或绘制示例），细节会随之自动调整。</translation>
    </message>
    <message>
        <source>New here? Our 5-minute tutorial walks you through a full detection, step by step.</source>
        <translation>第一次使用？我们 5 分钟的教程会带您逐步完成一次完整的检测。</translation>
    </message>
    <message>
        <source>Not Enough Disk Space</source>
        <translation>磁盘空间不足</translation>
    </message>
    <message>
        <source>Not enough credits: {n} tiles, only {left} left. Reduce the detail or zone, or subscribe.</source>
        <translation>积分不足：需要 {n} 个瓦片，仅剩 {left} 个。请降低细节或缩小区域，或订阅。</translation>
    </message>
    <message>
        <source>Not enough credits: {n} tiles, only {left} left. Reduce the detail or zone.</source>
        <translation>积分不足：需要 {n} 个瓦片，仅剩 {left} 个。请降低细节或缩小区域。</translation>
    </message>
    <message>
        <source>Online layer returned blank tiles for this area. The current zoom level may be outside the service's range, or this area has no coverage. Zoom to a level where the layer is visible on the map, then try again.</source>
        <translation>在线图层在此区域返回了空白瓦片。当前缩放级别可能超出该服务的支持范围，或此区域没有数据覆盖。请缩放到图层能在地图上显示的级别后重试。</translation>
    </message>
    <message>
        <source>Open the step-by-step tutorial</source>
        <translation>打开分步教程</translation>
    </message>
    <message>
        <source>Open the tutorial</source>
        <translation>打开教程</translation>
    </message>
    <message>
        <source>Opens terra-lab.ai in your browser.</source>
        <translation>在浏览器中打开 terra-lab.ai。</translation>
    </message>
    <message>
        <source>Opens your terra-lab.ai account in the browser.</source>
        <translation>在浏览器中打开您的 terra-lab.ai 账户。</translation>
    </message>
    <message>
        <source>Opens your terra-lab.ai dashboard in the browser.</source>
        <translation>在浏览器中打开您的 terra-lab.ai 控制面板。</translation>
    </message>
    <message>
        <source>Outline ONE example of the object on the map, then run again. Runs with a drawn example return far fewer empty results.</source>
        <translation>在地图上勾勒出该对象的一个示例，然后重新运行。带有绘制示例的运行结果为空的情况会大幅减少。</translation>
    </message>
    <message>
        <source>Planning AI Segmentation run</source>
        <translation>正在规划 AI Segmentation 运行</translation>
    </message>
    <message>
        <source>Pro plan</source>
        <translation>Pro 套餐</translation>
    </message>
    <message>
        <source>Remove downloaded AI data</source>
        <translation>移除已下载的 AI 数据</translation>
    </message>
    <message>
        <source>Remove the downloaded AI data from this computer?</source>
        <translation>要从此计算机中移除已下载的 AI 数据吗？</translation>
    </message>
    <message>
        <source>Removing...</source>
        <translation>正在移除...</translation>
    </message>
    <message>
        <source>Right level for {obj} in this zone.</source>
        <translation>该细节级别适合此区域内的{obj}。</translation>
    </message>
    <message>
        <source>Rotated raster</source>
        <translation>旋转栅格</translation>
    </message>
    <message>
        <source>Save {save} detections ({hidden} currently hidden by Confidence) to a layer before leaving?</source>
        <translation>离开前是否将 {save} 个检测结果（其中 {hidden} 个目前因置信度而被隐藏）保存到图层？</translation>
    </message>
    <message>
        <source>Save {save} detections to a layer before leaving?</source>
        <translation>离开前是否将 {save} 个检测结果保存到图层？</translation>
    </message>
    <message>
        <source>Segment library (view only)</source>
        <translation>分割对象库（仅可查看）</translation>
    </message>
    <message>
        <source>Segmentation failed. Please try again.</source>
        <translation>分割失败，请重试。</translation>
    </message>
    <message>
        <source>Sharper than {obj} usually needs - catches the smallest ones.</source>
        <translation>细节高于{obj}通常所需的水平，可捕捉到最小的对象。</translation>
    </message>
    <message>
        <source>Skipped {n} empty tiles (not charged).</source>
        <translation>已跳过 {n} 个空瓦片（不计费）。</translation>
    </message>
    <message>
        <source>Small {obj} may be missed at this level.</source>
        <translation>在此级别下，较小的{obj}可能会被遗漏。</translation>
    </message>
    <message>
        <source>Something went wrong saving your detections. Please try again.</source>
        <translation>保存检测结果时出错，请重试。</translation>
    </message>
    <message>
        <source>Something went wrong starting the detection. Please try again.</source>
        <translation>启动检测时出错，请重试。</translation>
    </message>
    <message>
        <source>Started at {pct}% - the usual sweet spot for this object type.</source>
        <translation>起始置信度为 {pct}%，这是该对象类型通常的最佳值。</translation>
    </message>
    <message>
        <source>Started at {pct}% to fit this run's scores - raise to tighten.</source>
        <translation>根据本次运行的评分，起始置信度为 {pct}%，提高数值可收紧结果。</translation>
    </message>
    <message>
        <source>Support code: {code}</source>
        <translation>支持代码：{code}</translation>
    </message>
    <message>
        <source>Team or organization?</source>
        <translation>团队或组织使用？</translation>
    </message>
    <message>
        <source>The AI service is waking up. Holding your spot…</source>
        <translation>AI 服务正在唤醒，正在为您保留名额…</translation>
    </message>
    <message>
        <source>The detection service is busy right now. Please try again in a moment.</source>
        <translation>检测服务目前繁忙，请稍后重试。</translation>
    </message>
    <message>
        <source>The service is temporarily unavailable (server error). Your connection is fine - please try again in a few minutes.</source>
        <translation>服务暂时不可用（服务器错误）。您的网络连接没有问题，请几分钟后再试。</translation>
    </message>
    <message>
        <source>There's a problem with your subscription. Open Settings to update your payment method or review your plan.</source>
        <translation>您的订阅出现问题，请打开设置以更新付款方式或查看您的套餐。</translation>
    </message>
    <message>
        <source>There's a problem with your subscription. Your last payment may have failed. Open your account to update your payment method or review your plan.</source>
        <translation>您的订阅出现问题，上次付款可能失败了。请打开您的账户以更新付款方式或查看您的套餐。</translation>
    </message>
    <message>
        <source>This deletes the local AI model files, signs you out, and resets the plugin. Your account and credits are not affected. Manual mode will download the files again next time you use it.</source>
        <translation>这将删除本地 AI 模型文件、退出登录并重置插件。您的账户和积分不受影响。下次使用手动模式时会重新下载文件。</translation>
    </message>
    <message>
        <source>This layer has no valid coordinate reference system. Set one in Layer Properties before detecting.</source>
        <translation>此图层没有有效的坐标参照系统。请在检测前于图层属性中进行设置。</translation>
    </message>
    <message>
        <source>This raster uses a geographic CRS (degrees), which distorts the imagery sent to the AI. For best results, reproject it to a projected CRS (e.g. UTM).</source>
        <translation>此栅格使用的是地理坐标系（度），会导致发送给 AI 的影像发生变形。为获得最佳效果，请将其重新投影为投影坐标系（例如 UTM）。</translation>
    </message>
    <message>
        <source>Tip: this raster has no overviews (pyramids). Build them (Raster menu, Miscellaneous, Build Overviews) to make detection much faster.</source>
        <translation>提示：此栅格没有概视图（金字塔）。生成概视图（栅格菜单 &gt; 杂项 &gt; 生成概视图）可大幅加快检测速度。</translation>
    </message>
    <message>
        <source>Try "{word}" instead</source>
        <translation>请改用“{word}”</translation>
    </message>
    <message>
        <source>Update now</source>
        <translation>立即更新</translation>
    </message>
    <message>
        <source>Update payment method</source>
        <translation>更新付款方式</translation>
    </message>
    <message>
        <source>Version {version} is available.</source>
        <translation>有新版本 {version} 可用。</translation>
    </message>
    <message>
        <source>Very fine for {obj} - large ones may come back split in parts.</source>
        <translation>对{obj}而言细节过高，较大的对象可能会被拆分成多个部分返回。</translation>
    </message>
    <message>
        <source>View detections as:</source>
        <translation>检测结果显示方式：</translation>
    </message>
    <message>
        <source>We read every message.</source>
        <translation>我们会认真阅读每一条消息。</translation>
    </message>
    <message>
        <source>Write to us:</source>
        <translation>联系我们：</translation>
    </message>
    <message>
        <source>Your reference</source>
        <translation>您的参考示例</translation>
    </message>
    <message>
        <source>confident</source>
        <translation>置信度高</translation>
    </message>
    <message>
        <source>polygons</source>
        <translation>多边形</translation>
    </message>
    <message>
        <source>uncertain</source>
        <translation>置信度低</translation>
    </message>
    <message>
        <source>upgrade to unlock it</source>
        <translation>升级以解锁</translation>
    </message>
    <message>
        <source>your object</source>
        <translation>您的对象</translation>
    </message>
    <message>
        <source>{n} found so far</source>
        <translation>目前已发现 {n} 个</translation>
    </message>
    <message>
        <source>{n} tiles could not be loaded from the layer server; results may be incomplete.</source>
        <translation>无法从图层服务器加载 {n} 个瓦片，结果可能不完整。</translation>
    </message>
    <message>
        <source>{n} tiles had no imagery and were not analyzed (not charged). Check the imagery layer loads over this area, then run Detect again.</source>
        <translation>{n} 个瓦片没有影像数据，未进行分析（不计费）。请确认影像图层在此区域能够正常加载，然后重新运行检测。</translation>
    </message>
    <message>
        <source>{n} {object} saved to layer "{name}"</source>
        <translation>已将 {n} 个{object}保存到图层“{name}”</translation>
    </message>
    <message>
        <source>{used} credits used</source>
        <translation>已使用 {used} 积分</translation>
    </message>
    <!-- v2.1.7 sync: append-to-layer export, singular forms, small-example guidance (2026-07-13) -->
    <message>
        <source>1 credit ~ 0.17 km² at default detail.</source>
        <translation>默认细节级别下，1 积分约对应 0.17 平方公里。</translation>
    </message>
    <message>
        <source>10,000 detections every month (~1,700 km²)</source>
        <translation>每月 10,000 次检测（约 1,700 平方公里）</translation>
    </message>
    <message>
        <source>Last run: {count} {object} exported · {area} km² · {used} credits used</source>
        <translation>上次运行：已 Export {count} 个 {object} · {area} 平方公里 · 消耗 {used} 积分</translation>
    </message>
    <message>
        <source>Last run: {count} {object} exported · {area} km² · {used} credits used, {left} left</source>
        <translation>上次运行：已 Export {count} 个 {object} · {area} 平方公里 · 消耗 {used} 积分，剩余 {left}</translation>
    </message>
    <message>
        <source>Last session: {count} polygon(s) exported · {area} km²</source>
        <translation>上次会话：已 Export {count} 个多边形 · {area} 平方公里</translation>
    </message>
    <message>
        <source>This zone is {area} km² - free trial zones go up to {max} km².</source>
        <translation>该区域面积为 {area} 平方公里，免费试用区域上限为 {max} 平方公里。</translation>
    </message>
    <message>
        <source>{area} km²</source>
        <translation>{area} 平方公里</translation>
    </message>
    <message>
        <source>Sending to the AI...</source>
        <translation>正在发送至 AI...</translation>
    </message>
    <message>
        <source>Spot reserved · starting in a few seconds...</source>
        <translation>名额已预留 · 数秒后开始...</translation>
    </message>
    <message>
        <source>Spot reserved · starting soon...</source>
        <translation>名额已预留 · 即将开始...</translation>
    </message>
    <message>
        <source>Stopping - keeping the tiles already found...</source>
        <translation>正在停止，已发现的瓦片将被保留...</translation>
    </message>
    <message>
        <source>Stopping...</source>
        <translation>正在停止...</translation>
    </message>
    <message>
        <source>The AI is starting up, almost there... {n}s</source>
        <translation>AI 正在启动，即将就绪... {n} 秒</translation>
    </message>
    <message>
        <source>Waking up the AI... {n}s</source>
        <translation>正在唤醒 AI... {n} 秒</translation>
    </message>
    <message>
        <source>You're next · starting now...</source>
        <translation>轮到您了 · 正在开始...</translation>
    </message>
    <message>
        <source>Cancelled</source>
        <translation>已取消</translation>
    </message>
    <message>
        <source>Free trial</source>
        <translation>免费试用</translation>
    </message>
    <message>
        <source>Select a raster layer to segment:</source>
        <translation>选择要分割的栅格图层：</translation>
    </message>
    <message>
        <source>Your {n} free detections are used up</source>
        <translation>您的 {n} 次免费检测已用完</translation>
    </message>
    <message>
        <source>1 object found</source>
        <translation>发现 1 个对象</translation>
    </message>
    <message>
        <source>Download AI model</source>
        <translation>下载 AI 模型</translation>
    </message>
    <message>
        <source>Export 1 polygon</source>
        <translation>Export 1 个多边形</translation>
    </message>
    <message>
        <source>Resolving object name</source>
        <translation>正在解析对象名称</translation>
    </message>
    <message>
        <source>This example is very small at the current detail level. Zoom the detail slider finer or draw a larger object.</source>
        <translation>此示例在当前细节级别下非常小。请将细节滑块调得更精细，或绘制更大的对象。</translation>
    </message>
    <message>
        <source>This example is very small even at the finest detail. Draw a larger object, or it may be too small to detect.</source>
        <translation>即使在最精细的细节级别下，此示例也非常小。请绘制更大的对象，否则可能太小而无法检测。</translation>
    </message>
    <message>
        <source>Your free detections are used up</source>
        <translation>您的免费检测已用完</translation>
    </message>
    <message>
        <source>&quot;{obj}&quot; is not an object the AI knows well. Drawing one example on the map shows it what to find.</source>
        <translation>“{obj}”不是 AI 熟悉的对象，在地图上绘制一个示例，让它知道要查找的内容。</translation>
    </message>
    <message>
        <source>&quot;{obj}&quot; is often missed from text alone. Draw one example on the map to find far more.</source>
        <translation>仅凭文字，“{obj}”很容易被遗漏。在地图上绘制一个示例可发现更多目标。</translation>
    </message>
    <message>
        <source>1 correction this round</source>
        <translation>本轮 1 次修正</translation>
    </message>
    <message>
        <source>1 detection removed</source>
        <translation>已移除 1 个检测结果</translation>
    </message>
    <message>
        <source>1 object</source>
        <translation>1 个对象</translation>
    </message>
    <message>
        <source>1 polygon added so far.</source>
        <translation>目前已添加 1 个多边形。</translation>
    </message>
    <message>
        <source>1 result</source>
        <translation>1 个结果</translation>
    </message>
    <message>
        <source>A detection run was interrupted before its results were exported. {n} detection(s) were saved.</source>
        <translation>检测运行被中断，此时结果尚未 Export，已保存 {n} 个检测结果。</translation>
    </message>
    <message>
        <source>A free run covers up to {cap} credits. Lower the detail or draw a smaller zone, or</source>
        <translation>免费运行最多可覆盖 {cap} 积分。请降低细节或缩小区域，或</translation>
    </message>
    <message>
        <source>A newer version of AI Segmentation is available with the latest fixes.</source>
        <translation>有新版本的 AI Segmentation 可用，包含最新修复。</translation>
    </message>
    <message>
        <source>A shape was removed. Click Save to confirm.</source>
        <translation>已移除一个形状，点击“保存”确认。</translation>
    </message>
    <message>
        <source>AI</source>
        <translation>AI</translation>
    </message>
    <message>
        <source>AI Environment Damaged</source>
        <translation>AI 环境已损坏</translation>
    </message>
    <message>
        <source>AI and Manual are two ways to fix the same polygon.</source>
        <translation>AI 和手动是修正同一个多边形的两种方式。</translation>
    </message>
    <message>
        <source>Add a missing polygon</source>
        <translation>添加遗漏的多边形</translation>
    </message>
    <message>
        <source>Add an object the AI missed. In AI, point at it and the on-device model outlines it, free; in Manual, draw its corners.</source>
        <translation>添加一个 AI 遗漏的对象。在 AI 模式下，点击它，本地模型即可免费勾出轮廓；在手动模式下，逐一绘制其节点。</translation>
    </message>
    <message>
        <source>Add another example - more references detect more</source>
        <translation>再添加一个示例，参考示例越多，检测越全</translation>
    </message>
    <message>
        <source>Add one more example for the best results.</source>
        <translation>再添加一个示例以获得最佳效果。</translation>
    </message>
    <message>
        <source>Add them to the project</source>
        <translation>将它们添加到项目中</translation>
    </message>
    <message>
        <source>Adding an object uses the free on-device AI, which is not installed yet. Install it now (a few minutes, in the background)? You can keep reviewing, and Add will arm automatically when it is ready.</source>
        <translation>添加对象使用免费的本地 AI，但尚未安装。现在安装吗？（需要几分钟，在后台进行）。您可以在此期间继续复核，安装完成后“添加”功能会自动启用。</translation>
    </message>
    <message>
        <source>Adding needs a one-time setup</source>
        <translation>添加功能需要进行一次性安装</translation>
    </message>
    <message>
        <source>Almost done - building the shapes...</source>
        <translation>即将完成，正在生成形状…</translation>
    </message>
    <message>
        <source>Almost there: in step 1, name the object your examples show - words plus examples detect best.</source>
        <translation>还差一步：在步骤 1 中为您示例所展示的对象命名，文字加示例的检测效果最佳。</translation>
    </message>
    <message>
        <source>Almost there: in step 2, outline one &quot;{word}&quot; on the map so the AI sees what yours look like.</source>
        <translation>还差一步：在步骤 2 中，在地图上圈出一个“{word}”，让 AI 了解它的样子。</translation>
    </message>
    <message>
        <source>Another QGIS window is installing the AI components. Wait for it to finish, then try again.</source>
        <translation>另一个 QGIS 窗口正在安装 AI 组件，请等待其完成后重试。</translation>
    </message>
    <message>
        <source>Another QGIS window is installing the AI engine. Wait for it to finish, then try again.</source>
        <translation>另一个 QGIS 窗口正在安装 AI 引擎，请等待其完成后重试。</translation>
    </message>
    <message>
        <source>Automatic detection failed</source>
        <translation>自动检测失败</translation>
    </message>
    <message>
        <source>Automatic: detect</source>
        <translation>自动模式：检测</translation>
    </message>
    <message>
        <source>Automatic: draw the zone</source>
        <translation>自动模式：绘制区域</translation>
    </message>
    <message>
        <source>Automatic: merge with neighbours</source>
        <translation>自动模式：与相邻对象合并</translation>
    </message>
    <message>
        <source>Automatic: review and Correct</source>
        <translation>自动模式：复核与修正</translation>
    </message>
    <message>
        <source>Best quality. Two references locked in.</source>
        <translation>最佳质量，已锁定两个参考示例。</translation>
    </message>
    <message>
        <source>Blocked by Antivirus or Security Software</source>
        <translation>被防病毒或安全软件拦截</translation>
    </message>
    <message>
        <source>Calculating...</source>
        <translation>正在计算…</translation>
    </message>
    <message>
        <source>Cancel the example box, the detection, or exit Automatic</source>
        <translation>取消示例框、取消检测，或退出自动模式</translation>
    </message>
    <message>
        <source>Cancel the merge</source>
        <translation>取消合并</translation>
    </message>
    <message>
        <source>Change recorded.</source>
        <translation>更改已记录。</translation>
    </message>
    <message>
        <source>Checking the object name</source>
        <translation>正在检查对象名称</translation>
    </message>
    <message>
        <source>Checking the object name...</source>
        <translation>正在检查对象名称…</translation>
    </message>
    <message>
        <source>Choose how to fix the polygon: AI points or QGIS vertices</source>
        <translation>选择修正多边形的方式：AI 点选还是 QGIS 节点编辑</translation>
    </message>
    <message>
        <source>Clean up the outlines</source>
        <translation>清理轮廓</translation>
    </message>
    <message>
        <source>Clear all</source>
        <translation>全部清空</translation>
    </message>
    <message>
        <source>Clear the points, then exit Automatic</source>
        <translation>清空点，然后退出自动模式</translation>
    </message>
    <message>
        <source>Clear the selection, or stop the segmentation</source>
        <translation>清空选区，或停止分割</translation>
    </message>
    <message>
        <source>Click a polygon, then click the spot the AI missed.</source>
        <translation>点击一个多边形，然后点击 AI 遗漏的位置。</translation>
    </message>
    <message>
        <source>Click a polygon, then drag any corner.</source>
        <translation>点击一个多边形，然后拖动任意节点。</translation>
    </message>
    <message>
        <source>Click an object on the map and the AI outlines it.</source>
        <translation>在地图上点击一个对象，AI 即可勾出其轮廓。</translation>
    </message>
    <message>
        <source>Click each corner of the object, then Finish.</source>
        <translation>依次点击对象的每个节点，然后点击“完成”。</translation>
    </message>
    <message>
        <source>Click each corner on the map, then Finish the line.</source>
        <translation>依次在地图上点击各个节点，然后完成这条线。</translation>
    </message>
    <message>
        <source>Click each piece of the object you want to join.</source>
        <translation>依次点击要拼接在一起的各个部分。</translation>
    </message>
    <message>
        <source>Click to open this step</source>
        <translation>点击展开此步骤</translation>
    </message>
    <message>
        <source>Close the fix, clear the selection, or exit the review</source>
        <translation>关闭修正、清空选区，或退出复核</translation>
    </message>
    <message>
        <source>Close the gaps inside this polygon, without filling the courtyards the rest of the layer is meant to keep.</source>
        <translation>闭合此多边形内部的缝隙，同时不填充图层其他部分需要保留的院落空间。</translation>
    </message>
    <message>
        <source>Close the line you are drawing. A right-click on the map does the same.</source>
        <translation>闭合正在绘制的线，在地图上右键点击效果相同。</translation>
    </message>
    <message>
        <source>Closes the hairline gaps between neighbouring shapes, for land cover maps.</source>
        <translation>闭合相邻形状之间的细小缝隙，适用于土地覆盖图。</translation>
    </message>
    <message>
        <source>Confirm the merge</source>
        <translation>确认合并</translation>
    </message>
    <message>
        <source>Correct</source>
        <translation>修正</translation>
    </message>
    <message>
        <source>Correct the polygons</source>
        <translation>修正多边形</translation>
    </message>
    <message>
        <source>Could not apply the new settings. Try a different value.</source>
        <translation>无法应用新设置，请尝试其他数值。</translation>
    </message>
    <message>
        <source>Could not check the AI components. See the log for details.</source>
        <translation>无法检查 AI 组件，详情请查看日志。</translation>
    </message>
    <message>
        <source>Could not load the saved run. The file may have been moved or deleted.</source>
        <translation>无法加载已保存的运行记录，文件可能已被移动或删除。</translation>
    </message>
    <message>
        <source>Credits come back on {date}</source>
        <translation>积分将于 {date} 恢复</translation>
    </message>
    <message>
        <source>Crop fetch was cancelled.</source>
        <translation>影像裁剪已取消。</translation>
    </message>
    <message>
        <source>Cut thin spurs off this polygon (0 = off). Raise it on a single ragged outline instead of eroding the whole layer.</source>
        <translation>削去此多边形上的细小毛刺（0 = 关闭）。请针对单个参差不齐的轮廓单独调高该值，而不是影响整个图层。</translation>
    </message>
    <message>
        <source>Delete this corner</source>
        <translation>删除此节点</translation>
    </message>
    <message>
        <source>Delete this polygon</source>
        <translation>删除此多边形</translation>
    </message>
    <message>
        <source>Delete this polygon (the Delete key works too, and a right-click on the map deletes the shape under the cursor). Undo brings it back.</source>
        <translation>删除此多边形（按 Delete 键同样有效，在地图上右键点击可删除光标下的形状）。撤销可将其恢复。</translation>
    </message>
    <message>
        <source>Deleting the downloaded data...</source>
        <translation>正在删除已下载的数据…</translation>
    </message>
    <message>
        <source>Dense area {current}/{total}</source>
        <translation>密集区域 {current}/{total}</translation>
    </message>
    <message>
        <source>Dense forest? &quot;Forest&quot; takes it as one block; &quot;Tree&quot; picks individual trees.</source>
        <translation>茂密的森林？“Forest”会作为一整块处理，“Tree”则会识别单株树木。</translation>
    </message>
    <message>
        <source>Detect with example only</source>
        <translation>仅使用示例检测</translation>
    </message>
    <message>
        <source>Detect with examples only</source>
        <translation>仅使用示例检测</translation>
    </message>
    <message>
        <source>Detect with text only</source>
        <translation>仅使用文字检测</translation>
    </message>
    <message>
        <source>Detection stopped early after {done} tile(s). The objects already found are kept below.</source>
        <translation>检测在处理 {done} 个瓦片后提前停止，已发现的对象保留在下方。</translation>
    </message>
    <message>
        <source>Discard reviewed results and run again? Confirm</source>
        <translation>放弃已复核的结果并重新运行？请确认</translation>
    </message>
    <message>
        <source>Distinct</source>
        <translation>分色</translation>
    </message>
    <message>
        <source>Drag a box around one look-alike to exclude.</source>
        <translation>拖出一个框，圈选一个要排除的相似对象。</translation>
    </message>
    <message>
        <source>Drag a box around one object.</source>
        <translation>拖出一个框，圈选一个对象。</translation>
    </message>
    <message>
        <source>Drag a corner to move it. Click an edge to add one, right-click removes.</source>
        <translation>拖动节点即可移动，点击边线可添加节点，右键点击可删除。</translation>
    </message>
    <message>
        <source>Drag a corner to move it. Double-click an edge to add one.</source>
        <translation>拖动节点即可移动，双击边线可添加节点。</translation>
    </message>
    <message>
        <source>Drag, add or delete the object&apos;s corners by hand.</source>
        <translation>手动拖动、添加或删除对象的节点。</translation>
    </message>
    <message>
        <source>Draw a line across the object to cut it into two.</source>
        <translation>在对象上绘制一条线，将其切分为两部分。</translation>
    </message>
    <message>
        <source>Draw a line right across the shape, then Finish.</source>
        <translation>绘制一条贯穿形状的线，然后点击“完成”。</translation>
    </message>
    <message>
        <source>Draw an example of the object first (step 2).</source>
        <translation>请先绘制该对象的示例（步骤 2）。</translation>
    </message>
    <message>
        <source>Draw an example, or type what to find.</source>
        <translation>绘制一个示例，或输入要查找的内容。</translation>
    </message>
    <message>
        <source>Draw its corners</source>
        <translation>绘制其节点</translation>
    </message>
    <message>
        <source>Draw one &apos;{object}&apos; - the AI finds the rest</source>
        <translation>绘制一个“{object}”，AI 会找出其余的</translation>
    </message>
    <message>
        <source>Draw one example - the AI finds the rest</source>
        <translation>绘制一个示例，AI 会找出其余的</translation>
    </message>
    <message>
        <source>Draw the new edge: start outside the shape, cross it, end outside, then Finish.</source>
        <translation>绘制新边线：从形状外部起笔，穿过形状，在外部收笔，然后点击“完成”。</translation>
    </message>
    <message>
        <source>Drawn examples</source>
        <translation>已绘制的示例</translation>
    </message>
    <message>
        <source>Drop points closer than this distance to a straight edge (0 = off). A distance, not a count: pushed high it can flatten curved walls. Points is usually the better dial for thinning an outline.</source>
        <translation>删除与直线边缘距离小于此值的节点（0 = 关闭）。这是一个距离，而非数量：调得过高会压平弯曲的墙面。细化轮廓时通常“节点数”是更合适的调节项。</translation>
    </message>
    <message>
        <source>Drop points closer than this distance to a straight edge (0 = off). A distance, not a count: pushed high it can flatten curved walls. Points is usually the better dial; this stays for comparison.</source>
        <translation>删除与直线边缘距离小于此值的节点（0 = 关闭）。这是一个距离，而非数量：调得过高会压平弯曲的墙面。通常“节点数”是更合适的调节项，此项保留仅作对比。</translation>
    </message>
    <message>
        <source>Drop this polygon&apos;s points closer than this distance to a straight edge (0 = off). A distance, not a count; Points is usually the better dial.</source>
        <translation>删除此多边形上与直线边缘距离小于此值的节点（0 = 关闭）。这是一个距离，而非数量；通常“节点数”是更合适的调节项。</translation>
    </message>
    <message>
        <source>Edit an existing polygon</source>
        <translation>编辑现有多边形</translation>
    </message>
    <message>
        <source>Export the polygons to a layer</source>
        <translation>将多边形 Export 到图层</translation>
    </message>
    <message>
        <source>Fewer points</source>
        <translation>减少节点数</translation>
    </message>
    <message>
        <source>Fill holes</source>
        <translation>填充孔洞</translation>
    </message>
    <message>
        <source>Fill only holes smaller than this ground area. Bigger holes (a road median, a courtyard) stay open. No limit = fill every hole.</source>
        <translation>仅填充地面面积小于此值的孔洞。更大的孔洞（如道路中央隔离带、院落）将保持开放。无限制 = 填充所有孔洞。</translation>
    </message>
    <message>
        <source>Finish the line</source>
        <translation>完成这条线</translation>
    </message>
    <message>
        <source>Fix method</source>
        <translation>修正方式</translation>
    </message>
    <message>
        <source>Give neighbouring shapes one exact border instead of a hairline gap or overlap. For land cover, where the map is one surface.</source>
        <translation>让相邻形状共用同一条精确边界，而非留下细小缝隙或重叠。适用于地图为单一连续表面的土地覆盖场景。</translation>
    </message>
    <message>
        <source>Give this one polygon its own shape settings, without moving the dials that drive the whole layer.</source>
        <translation>为这一个多边形单独设置形状参数，不影响驱动整个图层的全局设置。</translation>
    </message>
    <message>
        <source>Go back to picking polygons. Everything you kept stays, and so does the outline on screen.</source>
        <translation>返回选取多边形。您已保留的内容会保留，屏幕上的轮廓也会保留。</translation>
    </message>
    <message>
        <source>Go back to your zone, references and settings, then detect the whole zone again. Nothing is saved.</source>
        <translation>返回您的区域、参考示例和设置，然后重新检测整个区域。系统不会保存任何内容。</translation>
    </message>
    <message>
        <source>Grow / shrink</source>
        <translation>扩展 / 收缩</translation>
    </message>
    <message>
        <source>How detections are coloured on the map (visual only): Normal fill, Outline, Confidence heatmap, or a distinct colour per object to tell them apart.</source>
        <translation>检测结果在地图上的着色方式（仅影响显示）：普通填充、轮廓、置信度热力图，或每个对象使用不同颜色以便区分。</translation>
    </message>
    <message>
        <source>How many of each outline&apos;s points to keep, as a share. 100% keeps the class default; lower thins further, dropping the smallest detail first and keeping the corners and the surface. With Right angles on it runs first, so lowering it gives the squaring straight walls to work from.</source>
        <translation>保留每条轮廓节点的比例。100% 保持该类别的默认密度；数值越低，细化程度越高，优先去除最细小的细节，同时保留转角和整体轮廓。开启“直角化”时该项先执行，因此调低它可为直角化提供更平直的墙面基础。</translation>
    </message>
    <message>
        <source>How many of the outline&apos;s points to keep, as a share. 100% keeps the standard density; lower thins further, dropping the smallest detail first and keeping the corners and the surface. With Right angles on it runs first, so lowering it gives the squaring straight walls to work from.</source>
        <translation>保留轮廓节点的比例。100% 保持标准密度；数值越低，细化程度越高，优先去除最细小的细节，同时保留转角和整体轮廓。开启“直角化”时该项先执行，因此调低它可为直角化提供更平直的墙面基础。</translation>
    </message>
    <message>
        <source>How many of this polygon&apos;s points to keep. The count in the title row follows it. It runs before Right angles, so lowering it gives the squaring straight walls instead of a staircase.</source>
        <translation>此多边形要保留的节点数量，标题行中的数字会随之变化。该项在“直角化”之前执行，因此调低它可让直角化处理出平直的墙面，而不是锯齿状的台阶。</translation>
    </message>
    <message>
        <source>How sure the AI is about each object. Lower shows more, higher keeps only the sure ones.</source>
        <translation>AI 对每个对象的把握程度。数值越低显示越多，数值越高仅保留把握较大的对象。</translation>
    </message>
    <message>
        <source>Identify new shape</source>
        <translation>识别新形状</translation>
    </message>
    <message>
        <source>Installation Already Running</source>
        <translation>安装已在运行</translation>
    </message>
    <message>
        <source>Installation Path Problem</source>
        <translation>安装路径问题</translation>
    </message>
    <message>
        <source>Installation running in another window</source>
        <translation>安装正在另一个窗口中运行</translation>
    </message>
    <message>
        <source>Keep</source>
        <translation>保留</translation>
    </message>
    <message>
        <source>Keep these edits and go back to picking polygons.</source>
        <translation>保留这些编辑，返回选取多边形。</translation>
    </message>
    <message>
        <source>Keep this detection in Favorites</source>
        <translation>将此检测结果加入收藏</translation>
    </message>
    <message>
        <source>Keep this object in Favorites</source>
        <translation>将此对象加入收藏</translation>
    </message>
    <message>
        <source>Keep this one</source>
        <translation>保留此项</translation>
    </message>
    <message>
        <source>Keep this one, or click again to correct the outline.</source>
        <translation>保留此项，或再次点击以修正轮廓。</translation>
    </message>
    <message>
        <source>Keep this one, or keep placing corners.</source>
        <translation>保留此项，或继续放置节点。</translation>
    </message>
    <message>
        <source>Keep this outline and point at the next object. Shortcut: S</source>
        <translation>保留此轮廓，并点击下一个对象。快捷键：S</translation>
    </message>
    <message>
        <source>Left-click adds a keep point, right-click a trim point. The outline follows.</source>
        <translation>左键点击添加保留点，右键点击添加裁剪点，轮廓会随之更新。</translation>
    </message>
    <message>
        <source>Linux System Too Old</source>
        <translation>Linux 系统版本过旧</translation>
    </message>
    <message>
        <source>Loading stored detections ({done} of {total})</source>
        <translation>正在加载已保存的检测结果（{done}/{total}）</translation>
    </message>
    <message>
        <source>Maximum</source>
        <translation>最大值</translation>
    </message>
    <message>
        <source>Measuring AI data size...</source>
        <translation>正在测量 AI 数据大小…</translation>
    </message>
    <message>
        <source>Measuring the downloaded data...</source>
        <translation>正在测量已下载的数据…</translation>
    </message>
    <message>
        <source>Merge with neighbours</source>
        <translation>与相邻对象合并</translation>
    </message>
    <message>
        <source>Merge {n} shapes · Free</source>
        <translation>合并 {n} 个形状 · 免费</translation>
    </message>
    <message>
        <source>Minimum</source>
        <translation>最小值</translation>
    </message>
    <message>
        <source>Missing System Component</source>
        <translation>缺少系统组件</translation>
    </message>
    <message>
        <source>Move points</source>
        <translation>移动节点</translation>
    </message>
    <message>
        <source>Navigation (while a tool is armed)</source>
        <translation>导航（工具启用期间）</translation>
    </message>
    <message>
        <source>New polygon</source>
        <translation>新建多边形</translation>
    </message>
    <message>
        <source>New shape added. Click Save to keep it.</source>
        <translation>已添加新形状，点击“保存”以保留。</translation>
    </message>
    <message>
        <source>No connection to the sign-in service. Check your internet connection, then click Connect to try again.</source>
        <translation>无法连接登录服务，请检查网络连接，然后点击“连接”重试。</translation>
    </message>
    <message>
        <source>No detection in this zone. Try a more specific object word, or a finer detail level.</source>
        <translation>该区域内无检测结果，请尝试更具体的对象名称，或提高细节级别。</translation>
    </message>
    <message>
        <source>No detection under that click.</source>
        <translation>该点击位置未检测到对象。</translation>
    </message>
    <message>
        <source>No matches in this zone.</source>
        <translation>该区域内无匹配结果。</translation>
    </message>
    <message>
        <source>No object matches that search.</source>
        <translation>没有与该搜索匹配的对象。</translation>
    </message>
    <message>
        <source>Nothing changed. The line has to cross the outline twice, starting and ending outside the shape.</source>
        <translation>未发生更改。该线必须与轮廓相交两次，且起点和终点都在形状外部。</translation>
    </message>
    <message>
        <source>Nothing cleared the confidence bar in this zone.</source>
        <translation>该区域内没有对象达到置信度阈值。</translation>
    </message>
    <message>
        <source>Nothing was added. A polygon needs at least three corners.</source>
        <translation>未添加任何内容，多边形至少需要三个节点。</translation>
    </message>
    <message>
        <source>Nothing was split. The line has to cross the shape completely, starting and ending outside it.</source>
        <translation>未进行拆分。该线必须完全穿过形状，且起点和终点都在形状外部。</translation>
    </message>
    <message>
        <source>Now click the other pieces of this object.</source>
        <translation>现在点击此对象的其他部分。</translation>
    </message>
    <message>
        <source>One object came back split into several polygons. Click the others on the map, then confirm to merge them into one.</source>
        <translation>一个对象被拆分成了多个多边形返回。请在地图上点击其余部分，然后确认将它们合并为一个。</translation>
    </message>
    <message>
        <source>One object per run - Detect will run &quot;{first}&quot; first.</source>
        <translation>每次运行仅处理一个对象，检测将优先运行“{first}”。</translation>
    </message>
    <message>
        <source>One object per run - detecting &quot;{first}&quot; now. Run the other objects as separate detections.</source>
        <translation>每次运行仅处理一个对象，目前正在检测“{first}”。其他对象请作为单独的检测运行。</translation>
    </message>
    <message>
        <source>Only this polygon. Every other one follows the Shapes step.</source>
        <translation>仅此多边形；其他多边形仍遵循“形状”步骤的设置。</translation>
    </message>
    <message>
        <source>Open the selected saved polygon for AI editing</source>
        <translation>在 AI 编辑中打开所选的已保存多边形</translation>
    </message>
    <message>
        <source>Other</source>
        <translation>其他</translation>
    </message>
    <message>
        <source>Package Versions Conflict</source>
        <translation>软件包版本冲突</translation>
    </message>
    <message>
        <source>Pick a tool above, then edit the highlighted object.</source>
        <translation>在上方选择一个工具，然后编辑高亮的对象。</translation>
    </message>
    <message>
        <source>Pick at least two shapes to merge them.</source>
        <translation>至少选择两个形状才能合并。</translation>
    </message>
    <message>
        <source>Pick or un-pick an object</source>
        <translation>选中或取消选中一个对象</translation>
    </message>
    <message>
        <source>Point at it on the map</source>
        <translation>在地图上点击它</translation>
    </message>
    <message>
        <source>Points</source>
        <translation>节点数</translation>
    </message>
    <message>
        <source>Points the map back at this run with the same object and the same number of tiles, ready to detect. Nothing is spent until you do.</source>
        <translation>将地图重新定位到本次运行的位置，保持相同的对象和相同数量的瓦片，随时可以检测。在您操作之前不会消耗任何积分。</translation>
    </message>
    <message>
        <source>Points: {n}</source>
        <translation>节点数：{n}</translation>
    </message>
    <message>
        <source>Positive = grow outward, negative = shrink inward</source>
        <translation>正值 = 向外扩展，负值 = 向内收缩</translation>
    </message>
    <message>
        <source>Proxy Authentication Required</source>
        <translation>需要代理身份验证</translation>
    </message>
    <message>
        <source>Push this polygon&apos;s edge out (positive) or in (negative), for the one footprint the model cut short or overran.</source>
        <translation>将此多边形的边缘向外（正值）或向内（负值）推移，适用于模型描绘得过小或过大的单个对象轮廓。</translation>
    </message>
    <message>
        <source>QGIS could not activate the temporary review layer. Close any other editing session, then try Edit manually again.</source>
        <translation>QGIS 无法激活临时复核图层。请关闭其他编辑会话，然后重新尝试手动编辑。</translation>
    </message>
    <message>
        <source>QGIS could not open a browser. Open this address to finish signing in, then come back here:
{}</source>
        <translation>QGIS 无法打开浏览器。请打开以下地址完成登录，然后返回此处：
{}</translation>
    </message>
    <message>
        <source>QGIS could not open a browser. The sign-in address is copied to your clipboard: paste it into a browser to finish, then come back here.</source>
        <translation>QGIS 无法打开浏览器。登录地址已复制到剪贴板：请将其粘贴到浏览器中完成登录，然后返回此处。</translation>
    </message>
    <message>
        <source>QGIS could not save these edits. Fix the geometry and click Done again.</source>
        <translation>QGIS 无法保存这些编辑，请修复几何图形后重新点击“完成”。</translation>
    </message>
    <message>
        <source>Re-run the whole zone</source>
        <translation>重新运行整个区域</translation>
    </message>
    <message>
        <source>Reading the imagery around this polygon...</source>
        <translation>正在读取此多边形周边的影像…</translation>
    </message>
    <message>
        <source>Reading the imagery around your click...</source>
        <translation>正在读取点击位置周边的影像…</translation>
    </message>
    <message>
        <source>Reading this run...</source>
        <translation>正在读取本次运行…</translation>
    </message>
    <message>
        <source>Rebuilding shapes ({done} of {total})</source>
        <translation>正在重建形状（{done}/{total}）</translation>
    </message>
    <message>
        <source>Recovered {n} detection(s) to the layer {name}.</source>
        <translation>已将 {n} 个检测结果恢复到图层 {name}。</translation>
    </message>
    <message>
        <source>Redraw edge</source>
        <translation>重新绘制边线</translation>
    </message>
    <message>
        <source>Remove the corner you picked. The Delete key does the same.</source>
        <translation>删除您选中的节点，Delete 键效果相同。</translation>
    </message>
    <message>
        <source>Remove the selected detection</source>
        <translation>移除所选的检测结果</translation>
    </message>
    <message>
        <source>Removing the downloaded AI data. This window closes when it is done.</source>
        <translation>正在移除已下载的 AI 数据，完成后此窗口将关闭。</translation>
    </message>
    <message>
        <source>Removing the downloaded AI data...</source>
        <translation>正在移除已下载的 AI 数据…</translation>
    </message>
    <message>
        <source>Replace one side by drawing a new line across the outline.</source>
        <translation>绘制一条穿过轮廓的新线，以替换其中一侧。</translation>
    </message>
    <message>
        <source>Report this problem</source>
        <translation>报告此问题</translation>
    </message>
    <message>
        <source>Reset to shared</source>
        <translation>重置为共享设置</translation>
    </message>
    <message>
        <source>Reshape needs a one-time setup</source>
        <translation>重塑功能需要进行一次性安装</translation>
    </message>
    <message>
        <source>Reshaping uses the free on-device AI, which is not installed yet. Install it now (a few minutes, in the background)? You can keep reviewing, and reshaping will open automatically when it is ready.</source>
        <translation>重塑功能使用免费的本地 AI，但尚未安装。现在安装吗？（需要几分钟，在后台进行）。您可以在此期间继续复核，安装完成后重塑面板会自动打开。</translation>
    </message>
    <message>
        <source>Right angles</source>
        <translation>直角化</translation>
    </message>
    <message>
        <source>Round corners</source>
        <translation>圆角</translation>
    </message>
    <message>
        <source>Round corners for natural shapes like trees and bushes. Lower Points for smoother results.</source>
        <translation>圆角适用于树木、灌木等自然形状。降低“节点数”可获得更平滑的效果。</translation>
    </message>
    <message>
        <source>Round this polygon&apos;s corners, for a tree or a pond among squared neighbours.</source>
        <translation>为此多边形做圆角处理，适用于方正邻边中的树木或池塘等对象。</translation>
    </message>
    <message>
        <source>Run the detection</source>
        <translation>运行检测</translation>
    </message>
    <message>
        <source>Run this zone again</source>
        <translation>重新运行此区域</translation>
    </message>
    <message>
        <source>Running low: {n} free detections left, back on {date}. &lt;a href=&quot;{url}&quot;&gt;Subscribe&lt;/a&gt; to keep going.</source>
        <translation>即将用完：还剩 {n} 次免费检测，将于 {date} 恢复。&lt;a href=&quot;{url}&quot;&gt;订阅&lt;/a&gt;以继续使用。</translation>
    </message>
    <message>
        <source>Same setup as your last run - the result will match. Add an example or change the detail for a different result.</source>
        <translation>与上次运行的设置相同，结果也会相同。添加示例或更改细节以获得不同的结果。</translation>
    </message>
    <message>
        <source>Save</source>
        <translation>保存</translation>
    </message>
    <message>
        <source>Save the fix and go back to the review</source>
        <translation>保存修正并返回复核</translation>
    </message>
    <message>
        <source>Save this polygon and go back to picking.</source>
        <translation>保存此多边形并返回选取。</translation>
    </message>
    <message>
        <source>Setting up the on-device AI in the background. You can keep reviewing; reshaping opens automatically when it is ready.</source>
        <translation>正在后台设置本地 AI。您可以继续复核，准备就绪后重塑面板会自动打开。</translation>
    </message>
    <message>
        <source>Setting up the on-device AI in the background...</source>
        <translation>正在后台设置本地 AI…</translation>
    </message>
    <message>
        <source>Settings for this polygon</source>
        <translation>此多边形的设置</translation>
    </message>
    <message>
        <source>Shadows getting detected instead of trees? Draw an &apos;Exclude a look-alike&apos; box on one shadow - the AI drops similar false positives.</source>
        <translation>检测到的是阴影而不是树木？在一处阴影上绘制“排除相似对象”框，AI 会剔除类似的误检结果。</translation>
    </message>
    <message>
        <source>Shape updated. Keep editing, or click Save.</source>
        <translation>形状已更新，继续编辑，或点击“保存”。</translation>
    </message>
    <message>
        <source>Shapes</source>
        <translation>形状</translation>
    </message>
    <message>
        <source>Shared borders:</source>
        <translation>共享边界：</translation>
    </message>
    <message>
        <source>Shave thin spikes and ragged bits off each shape&apos;s outline. It leaves the main body alone; higher values trim more. 0 = off.</source>
        <translation>削去每个形状轮廓上的细小尖刺和参差不齐的部分，不影响主体；数值越高，修剪越多。0 = 关闭。</translation>
    </message>
    <message>
        <source>Show what it looks like</source>
        <translation>查看效果</translation>
    </message>
    <message>
        <source>Simplify</source>
        <translation>简化</translation>
    </message>
    <message>
        <source>Snap walls to right angles, 45 degree walls included. Made for buildings, pools and solar panels. A shape it would distort is left as it is.</source>
        <translation>将墙面吸附为直角，包括 45 度墙面。适用于建筑、泳池和太阳能板；若吸附会使形状失真，则保持原样。</translation>
    </message>
    <message>
        <source>Something went wrong preparing the results. Please run Detect again.</source>
        <translation>准备结果时出现问题，请重新运行检测。</translation>
    </message>
    <message>
        <source>Something went wrong preparing the results. Your detections were saved to the layer {name}.</source>
        <translation>准备结果时出现问题，您的检测结果已保存到图层 {name}。</translation>
    </message>
    <message>
        <source>Split</source>
        <translation>拆分</translation>
    </message>
    <message>
        <source>Square this polygon&apos;s edges, or leave them as traced while the rest of the layer stays squared.</source>
        <translation>将此多边形的边缘变为直角，或保持原始描绘形态，同时图层其余部分仍为直角化处理。</translation>
    </message>
    <message>
        <source>Star a detection or an object to keep it here.</source>
        <translation>收藏检测结果或对象，将其保留在此处。</translation>
    </message>
    <message>
        <source>Still waiting for the sign-in page. If no browser opened, or the page shows an error, click Cancel and try again.</source>
        <translation>仍在等待登录页面。如果没有打开浏览器，或页面显示错误，请点击“取消”后重试。</translation>
    </message>
    <message>
        <source>Stop adding</source>
        <translation>停止添加</translation>
    </message>
    <message>
        <source>Stopping the local AI...</source>
        <translation>正在停止本地 AI…</translation>
    </message>
    <message>
        <source>Switch to AI to shape this polygon on its own.</source>
        <translation>切换到 AI 模式，让其自动为此多边形塑形。</translation>
    </message>
    <message>
        <source>That area does not touch the object you are editing, so nothing was added. Reshaping works on one object at a time.</source>
        <translation>该区域未与您正在编辑的对象相接触，因此未添加任何内容。重塑每次只能处理一个对象。</translation>
    </message>
    <message>
        <source>That ground belongs to another object, so nothing was added. Edit that object instead, or join the two with Merge with neighbours.</source>
        <translation>该区域属于另一个对象，因此未添加任何内容。请改为编辑该对象，或使用“与相邻对象合并”将两者拼接。</translation>
    </message>
    <message>
        <source>That is another object. The one you were editing is saved, and this one is now selected.</source>
        <translation>这是另一个对象。您正在编辑的对象已保存，当前已选中此对象。</translation>
    </message>
    <message>
        <source>The AI finds every object that looks like your examples - you can draw up to 3.</source>
        <translation>AI 会找出所有与您示例相似的对象，最多可绘制 3 个示例。</translation>
    </message>
    <message>
        <source>The AI outlines it, free, on your computer.</source>
        <translation>AI 会在您的电脑上免费为其勾出轮廓。</translation>
    </message>
    <message>
        <source>The connection to the server was interrupted. Please try again.</source>
        <translation>与服务器的连接已中断，请重试。</translation>
    </message>
    <message>
        <source>The detection service had a problem and the run stopped. Please try again.</source>
        <translation>检测服务出现问题，运行已停止，请重试。</translation>
    </message>
    <message>
        <source>The detection stopped responding. Keeping the {n} tiles already found.</source>
        <translation>检测服务已无响应，将保留已发现的 {n} 个瓦片。</translation>
    </message>
    <message>
        <source>The imagery reader could not be loaded, and repairing the installation did not fix it. Please report this so we can look into it.

{details}</source>
        <translation>无法加载影像读取器，修复安装也未能解决问题。请报告此问题以便我们进行排查。

{details}</translation>
    </message>
    <message>
        <source>The installer could not start a helper process (a damaged Python launcher). Click Reinstall Dependencies to rebuild the environment from scratch.</source>
        <translation>安装程序无法启动辅助进程（Python 启动器已损坏）。请点击“重新安装依赖项”以从头重建环境。</translation>
    </message>
    <message>
        <source>The pieces do not touch, so the result is one object in several parts.</source>
        <translation>这些部分互不相邻，因此结果是一个分为多段的对象。</translation>
    </message>
    <message>
        <source>The removal could not start. You are signed out, but the downloaded AI data is still on this computer. Try again.</source>
        <translation>移除操作无法启动。您已退出登录，但已下载的 AI 数据仍保留在本机上，请重试。</translation>
    </message>
    <message>
        <source>The removal did not finish. Close this window, then check the AI data folder before trying again.</source>
        <translation>移除操作未能完成。请关闭此窗口，检查 AI 数据文件夹后再重试。</translation>
    </message>
    <message>
        <source>The removal is already running.</source>
        <translation>移除操作已在进行中。</translation>
    </message>
    <message>
        <source>The reply did not come from the service. If this network shows a sign-in page, open it in your browser first, then try again.</source>
        <translation>收到的回复并非来自该服务。如果此网络会显示登录页面，请先在浏览器中打开该页面，然后重试。</translation>
    </message>
    <message>
        <source>The server returned an unexpected response. Please try again.</source>
        <translation>服务器返回了意外的响应，请重试。</translation>
    </message>
    <message>
        <source>Thin this polygon&apos;s points before you edit them by hand. 100% keeps the outline as it is.</source>
        <translation>在手动编辑之前细化此多边形的节点。100% 表示保持轮廓不变。</translation>
    </message>
    <message>
        <source>This image has no position on the map, so Automatic cannot place what it finds. Give it one with the QGIS Georeferencer, or use Manual mode on it as is.</source>
        <translation>此图像在地图上没有地理位置，因此自动模式无法定位其检测结果。请使用 QGIS 地理配准器为其添加位置，或直接以当前状态使用手动模式。</translation>
    </message>
    <message>
        <source>This polygon</source>
        <translation>此多边形</translation>
    </message>
    <message>
        <source>This raster has no coordinate reference system, so polygons will use pixel coordinates. Set a CRS in Layer Properties for georeferenced output.</source>
        <translation>此栅格没有坐标参照系统，多边形将使用像素坐标。请在图层属性中设置 CRS 以获得具有地理参照的输出。</translation>
    </message>
    <message>
        <source>This raster is rotated. Run Warp (Reproject) on it to straighten it before segmenting.</source>
        <translation>此栅格已旋转，请在分割前运行 Warp（重投影）将其校正。</translation>
    </message>
    <message>
        <source>This raster is rotated. Run Warp (Reproject) on it to straighten it first. Manual mode cannot read it either.</source>
        <translation>此栅格已旋转，请先运行 Warp（重投影）将其校正。手动模式同样无法读取该栅格。</translation>
    </message>
    <message>
        <source>This run costs more credits than a free run covers. Lower the detail or draw a smaller zone, or</source>
        <translation>本次运行所需积分超出免费额度。请降低细节或缩小区域，或</translation>
    </message>
    <message>
        <source>This run did not keep where it looked, so it cannot be pointed at the same place. Draw the zone again.</source>
        <translation>本次运行未保留其检测位置，因此无法重新定位到同一位置，请重新绘制区域。</translation>
    </message>
    <message>
        <source>Those shapes could not be joined. Nothing was changed.</source>
        <translation>这些形状无法拼接，未作任何更改。</translation>
    </message>
    <message>
        <source>Trim spikes</source>
        <translation>修剪尖刺</translation>
    </message>
    <message>
        <source>Two references give the strongest detection. Draw a second to reach best quality.</source>
        <translation>两个参考示例可获得最强的检测效果，请再绘制一个以达到最佳质量。</translation>
    </message>
    <message>
        <source>Type a word for the object first (step 1).</source>
        <translation>请先为该对象输入一个词（步骤 1）。</translation>
    </message>
    <message>
        <source>Unavailable while Right angles is on. Turn it off to adjust this setting.</source>
        <translation>开启“直角化”时不可用，请关闭它以调整此设置。</translation>
    </message>
    <message>
        <source>Undo</source>
        <translation>撤销</translation>
    </message>
    <message>
        <source>Undo last</source>
        <translation>撤销上一步</translation>
    </message>
    <message>
        <source>Undo the last correction</source>
        <translation>撤销上一次修正</translation>
    </message>
    <message>
        <source>Undo the last thing you did here: the point you just placed, or the last edit.</source>
        <translation>撤销您在此处的最后一步操作：刚放置的点，或最近一次编辑。</translation>
    </message>
    <message>
        <source>Unsupported Mac and Python Combination</source>
        <translation>不支持的 Mac 与 Python 组合</translation>
    </message>
    <message>
        <source>Up to</source>
        <translation>最多</translation>
    </message>
    <message>
        <source>Writing the file...</source>
        <translation>正在写入文件…</translation>
    </message>
    <message>
        <source>You place the corners, the same as on any QGIS layer.</source>
        <translation>由您放置节点，与在任何 QGIS 图层中一样。</translation>
    </message>
    <message>
        <source>Your examples drive the search - naming the object makes it even more accurate.</source>
        <translation>您的示例将引导搜索，为对象命名可进一步提高准确度。</translation>
    </message>
    <message>
        <source>Your free detections come back on {date}.</source>
        <translation>您的免费检测将于 {date} 恢复。</translation>
    </message>
    <message>
        <source>Your network proxy requires a username and password. Enter them in QGIS &gt; Settings &gt; Options &gt; Network, then restart QGIS and try again.</source>
        <translation>您的网络代理需要用户名和密码。请在 QGIS &gt; 设置 &gt; 选项 &gt; 网络 中输入，然后重启 QGIS 并重试。</translation>
    </message>
    <message>
        <source>a month ago</source>
        <translation>一个月前</translation>
    </message>
    <message>
        <source>a week ago</source>
        <translation>一周前</translation>
    </message>
    <message>
        <source>a year ago</source>
        <translation>一年前</translation>
    </message>
    <message>
        <source>applies to every polygon</source>
        <translation>适用于所有多边形</translation>
    </message>
    <message>
        <source>fine-tune the edges</source>
        <translation>微调边缘</translation>
    </message>
    <message>
        <source>hide anything outside this range</source>
        <translation>隐藏此范围之外的内容</translation>
    </message>
    <message>
        <source>how each outline is styled</source>
        <translation>各轮廓的样式设置方式</translation>
    </message>
    <message>
        <source>how the outline is styled</source>
        <translation>轮廓的样式设置方式</translation>
    </message>
    <message>
        <source>{count} polygons added so far.</source>
        <translation>目前已添加 {count} 个多边形。</translation>
    </message>
    <message>
        <source>{count} vertices</source>
        <translation>{count} 个节点</translation>
    </message>
    <message>
        <source>{credits} credits</source>
        <translation>{credits} 积分</translation>
    </message>
    <message>
        <source>{hidden} hidden by the filters</source>
        <translation>{hidden} 个已被筛选条件隐藏</translation>
    </message>
    <message>
        <source>{n} &quot;{object}&quot; found so far</source>
        <translation>目前已找到 {n} 个“{object}”</translation>
    </message>
    <message>
        <source>{n} corrections this round</source>
        <translation>本轮 {n} 次修正</translation>
    </message>
    <message>
        <source>{n} kept</source>
        <translation>已保留 {n} 个</translation>
    </message>
    <message>
        <source>{n} months ago</source>
        <translation>{n} 个月前</translation>
    </message>
    <message>
        <source>{n} objects</source>
        <translation>{n} 个对象</translation>
    </message>
    <message>
        <source>{n} point placed.</source>
        <translation>已放置 {n} 个点。</translation>
    </message>
    <message>
        <source>{n} points placed.</source>
        <translation>已放置 {n} 个点。</translation>
    </message>
    <message>
        <source>{n} results</source>
        <translation>{n} 个结果</translation>
    </message>
    <message>
        <source>{n} shape(s) edited this session</source>
        <translation>本次会话已编辑 {n} 个形状</translation>
    </message>
    <message>
        <source>{n} shapes merged into one · Free</source>
        <translation>{n} 个形状已合并为一个 · 免费</translation>
    </message>
    <message>
        <source>{n} shapes picked. Press Enter to join.</source>
        <translation>已选择 {n} 个形状，按 Enter 拼接。</translation>
    </message>
    <message>
        <source>{n} tile(s) took too long to load and are missing from this result.</source>
        <translation>{n} 个瓦片加载耗时过长，未包含在本次结果中。</translation>
    </message>
    <message>
        <source>{n} weeks ago</source>
        <translation>{n} 周前</translation>
    </message>
    <message>
        <source>{n} years ago</source>
        <translation>{n} 年前</translation>
    </message>
    <message>
        <source>{tiles} tiles</source>
        <translation>{tiles} 个瓦片</translation>
    </message>
    <!-- Correct step rework, install and precision wording (2026-07-30) -->
    <message>
        <source>1 credit ~ 0.17 km² at default precision.</source>
        <translation>默认精度下，1 积分约对应 0.17 平方公里。</translation>
    </message>
    <message>
        <source>10,000 credits a month.</source>
        <translation>每月 10,000 积分。</translation>
    </message>
    <message>
        <source>A free run covers up to {cap} credits. This one costs more.</source>
        <translation>免费运行最多可覆盖 {cap} 积分。本次运行消耗更多积分。</translation>
    </message>
    <message>
        <source>Adding an object uses the free on-device AI, which is not installed yet. Install it now? It runs once and takes a few minutes. The review waits for it, then arms Add for you.</source>
        <translation>添加对象使用免费的本地 AI，但尚未安装。现在安装吗？只需运行一次，需要几分钟。复核会等待安装完成，然后自动为您启用“添加”。</translation>
    </message>
    <message>
        <source>At this precision {obj} is too small to spot - raise the precision.</source>
        <translation>在当前精度下，{obj}太小，难以发现，请提高精度。</translation>
    </message>
    <message>
        <source>Automatic mode is ready. The on-device AI could not be installed, so Manual mode and the AI fix are off until it is. Everything else works.</source>
        <translation>自动模式已就绪。本地 AI 未能安装，因此手动模式和 AI 修正在安装完成前不可用。其余功能均可正常使用。</translation>
    </message>
    <message>
        <source>Automatic mode needs the AI environment, and it is missing or incomplete. Install the dependencies from the plugin panel, then run Detect again.</source>
        <translation>自动模式需要 AI 环境，但该环境缺失或不完整。请在插件面板中安装依赖项，然后重新运行检测。</translation>
    </message>
    <message>
        <source>Automatic mode scans your zone tile by tile. 1 tile = 1 credit, so this run costs about {n} credits. More precision splits the zone into more tiles, which costs more credits.</source>
        <translation>自动模式会逐个瓦片扫描您的区域。1 个瓦片 = 1 积分，因此本次运行大约消耗 {n} 积分。精度越高，区域被划分的瓦片越多，消耗的积分也越多。</translation>
    </message>
    <message>
        <source>Cancel setup</source>
        <translation>取消安装</translation>
    </message>
    <message>
        <source>Clean up this outline</source>
        <translation>清理此轮廓</translation>
    </message>
    <message>
        <source>Click an object and the AI outlines it. You check and save each polygon yourself, one at a time. Use Automatic mode to get every object in a zone at once.</source>
        <translation>点击一个对象，AI 即可为其勾出轮廓。您可以逐一亲自检查并保存每个多边形。使用自动模式可一次性获取区域内的所有对象。</translation>
    </message>
    <message>
        <source>Click points around one look-alike, then double-click to close.</source>
        <translation>在一个相似对象周围逐点点击，然后双击以闭合。</translation>
    </message>
    <message>
        <source>Click points around one object, then double-click to close.</source>
        <translation>在一个对象周围逐点点击，然后双击以闭合。</translation>
    </message>
    <message>
        <source>Click the layer name to see it on the map</source>
        <translation>点击图层名称即可在地图上查看</translation>
    </message>
    <message>
        <source>Click to open your dashboard</source>
        <translation>点击打开您的控制面板</translation>
    </message>
    <message>
        <source>Connection is slow - still working, tiles already found are kept...</source>
        <translation>连接较慢，仍在处理中，已发现的瓦片将被保留...</translation>
    </message>
    <message>
        <source>Could not save your detections to a file.</source>
        <translation>无法将您的检测结果保存到文件。</translation>
    </message>
    <message>
        <source>Downloads Blocked by Your Network</source>
        <translation>下载被您的网络阻止</translation>
    </message>
    <message>
        <source>Draw a smaller zone, or &lt;a href=&quot;{url}&quot;&gt;Upgrade to Pro&lt;/a&gt; to segment areas of any size.</source>
        <translation>请绘制更小的区域，或&lt;a href=&quot;{url}&quot;&gt;升级到 Pro&lt;/a&gt;以分割任意大小的区域。</translation>
    </message>
    <message>
        <source>Draw a zone, name one kind of object, and get all of them in one run. Use Manual mode to work one object at a time.</source>
        <translation>绘制一个区域，命名一种对象，即可在一次运行中获取全部结果。使用手动模式可逐一处理对象。</translation>
    </message>
    <message>
        <source>Draw another example</source>
        <translation>再绘制一个示例</translation>
    </message>
    <message>
        <source>Drawing (click to stop)</source>
        <translation>正在绘制（点击以停止）</translation>
    </message>
    <message>
        <source>Errors, versions and the words you type, linked to your account. Never your imagery, layers or coordinates.</source>
        <translation>错误信息、版本号和您输入的文字，均与您的账户关联。绝不包含您的影像、图层或坐标信息。</translation>
    </message>
    <message>
        <source>Fixing a polygon uses the free on-device AI, which is not installed yet. Install it now? It runs once and takes a few minutes. The review waits for it, then opens this polygon for you.</source>
        <translation>修正多边形使用免费的本地 AI，但尚未安装。现在安装吗？只需运行一次，需要几分钟。复核会等待安装完成，然后自动为您打开此多边形。</translation>
    </message>
    <message>
        <source>Fixing needs a one-time setup</source>
        <translation>修正功能需要进行一次性安装</translation>
    </message>
    <message>
        <source>Helps us fix bugs faster.</source>
        <translation>帮助我们更快地修复问题。</translation>
    </message>
    <message>
        <source>Higher precision splits the zone into more tiles. Each tile costs 1 credit and captures smaller objects.</source>
        <translation>更高的精度会将区域划分为更多瓦片。每个瓦片消耗 1 积分，并能捕捉更小的对象。</translation>
    </message>
    <message>
        <source>Keep detecting without limits:</source>
        <translation>不受限制地持续检测：</translation>
    </message>
    <message>
        <source>Keep installing</source>
        <translation>继续安装</translation>
    </message>
    <message>
        <source>Keep this shape. The polygon stays picked, so you can still adjust, merge or delete it.</source>
        <translation>保留此形状。该多边形将保持选中，您仍可以调整、合并或删除它。</translation>
    </message>
    <message>
        <source>Last run: {n} {object}</source>
        <translation>上次运行：{n} 个{object}</translation>
    </message>
    <message>
        <source>Last run: {n} {object} in {layer}</source>
        <translation>上次运行：{n} 个{object}，位于{layer}</translation>
    </message>
    <message>
        <source>Loading the imagery...</source>
        <translation>正在加载影像...</translation>
    </message>
    <message>
        <source>Loading the imagery... {n}s</source>
        <translation>正在加载影像... {n}s</translation>
    </message>
    <message>
        <source>Max precision for this zone - draw a larger zone to go finer.</source>
        <translation>该区域已达到最高精度，绘制更大的区域可获得更精细的结果。</translation>
    </message>
    <message>
        <source>More precision cuts the zone into more tiles and costs more credits.</source>
        <translation>精度越高，区域被划分的瓦片越多，消耗的积分也越多。</translation>
    </message>
    <message>
        <source>More precision finds smaller objects.</source>
        <translation>精度越高，能发现的对象越小。</translation>
    </message>
    <message>
        <source>More precision keeps helping {obj} in this zone.</source>
        <translation>在此区域中，提高精度仍能持续改善{obj}的检测效果。</translation>
    </message>
    <message>
        <source>Name the object (or draw an example) first - Precision then tunes itself to it.</source>
        <translation>请先命名对象（或绘制示例），精度会随之自动调整。</translation>
    </message>
    <message>
        <source>Next: clean up the outlines</source>
        <translation>下一步：清理轮廓</translation>
    </message>
    <message>
        <source>Next: fix what looks wrong</source>
        <translation>下一步：修正看起来不对的地方</translation>
    </message>
    <message>
        <source>No detection in this zone. Try a more specific object word, or more precision.</source>
        <translation>该区域内无检测结果，请尝试更具体的对象名称，或提高精度。</translation>
    </message>
    <message>
        <source>No image over this zone at this precision, so nothing was analyzed (not charged). Lower Precision, or pick a layer that covers this area.</source>
        <translation>在当前精度下，该区域没有影像覆盖，因此未进行分析（不计费）。请降低精度，或选择一个覆盖此区域的图层。</translation>
    </message>
    <message>
        <source>Not enough credits: {n} tiles, only {left} left. Lower the precision or the zone.</source>
        <translation>积分不足：需要 {n} 个瓦片，仅剩 {left} 个。请降低精度或缩小区域。</translation>
    </message>
    <message>
        <source>Opens your terra-lab.ai dashboard: your plan, your credits and your payment details.</source>
        <translation>打开您的 terra-lab.ai 控制面板：您的套餐、积分和付款信息。</translation>
    </message>
    <message>
        <source>Outline settings</source>
        <translation>轮廓设置</translation>
    </message>
    <message>
        <source>Precision</source>
        <translation>精度</translation>
    </message>
    <message>
        <source>Ready for Automatic mode</source>
        <translation>自动模式已就绪</translation>
    </message>
    <message>
        <source>Running low: {n} free detections left, back on {date}. &lt;a href=&quot;{url}&quot;&gt;Upgrade to Pro&lt;/a&gt; to keep going.</source>
        <translation>即将用完：还剩 {n} 次免费检测，将于 {date} 恢复。&lt;a href=&quot;{url}&quot;&gt;升级到 Pro&lt;/a&gt;以继续使用。</translation>
    </message>
    <message>
        <source>Running low: {n} free detections left. &lt;a href=&quot;{url}&quot;&gt;Upgrade to Pro&lt;/a&gt; to keep going.</source>
        <translation>免费次数即将用完：剩余 {n} 次免费检测。&lt;a href=&quot;{url}&quot;&gt;升级到 Pro&lt;/a&gt;以继续使用。</translation>
    </message>
    <message>
        <source>Same setup as your last run - the result will match. Add an example or change the precision for a different result.</source>
        <translation>与上次运行的设置相同，结果也会相同。添加示例或更改精度以获得不同的结果。</translation>
    </message>
    <message>
        <source>Saved to {layer}</source>
        <translation>已保存到 {layer}</translation>
    </message>
    <message>
        <source>Setting up the on-device AI. This runs once and takes a few minutes. The review waits here until it is done.</source>
        <translation>正在设置本地 AI。此过程只需运行一次，需要几分钟。复核将在此等待，直到设置完成。</translation>
    </message>
    <message>
        <source>Setting up the on-device AI...</source>
        <translation>正在设置本地 AI...</translation>
    </message>
    <message>
        <source>Shadows getting detected instead of trees? Use 'Exclude a look-alike' on one shadow - the AI drops similar false positives.</source>
        <translation>检测到的是阴影而不是树木？在一处阴影上使用“排除相似对象”，AI 会剔除类似的误检结果。</translation>
    </message>
    <message>
        <source>Share of each outline's points to keep. 100% is the class default.
Lower thins the smallest detail first, keeps the corners, and gives Right angles straight walls to square.</source>
        <translation>保留每条轮廓节点的比例。100% 为该类别的默认值。
调低时优先去除最细小的细节，同时保留转角，并为“直角化”提供可拉直的墙面。</translation>
    </message>
    <message>
        <source>Share of the outline's points to keep. 100% is the standard density.
Lower thins the smallest detail first, keeps the corners, and gives Right angles straight walls to square.</source>
        <translation>保留轮廓节点的比例。100% 为标准密度。
调低时优先去除最细小的细节，同时保留转角，并为“直角化”提供可拉直的墙面。</translation>
    </message>
    <message>
        <source>Share usage statistics with TerraLab</source>
        <translation>与 TerraLab 分享使用统计信息</translation>
    </message>
    <message>
        <source>Simplify this outline first</source>
        <translation>先简化此轮廓</translation>
    </message>
    <message>
        <source>Stop the setup</source>
        <translation>停止安装</translation>
    </message>
    <message>
        <source>Stop the setup and go back to the review. The AI fix stays unavailable until you install it.</source>
        <translation>停止安装并返回复核。在您完成安装前，AI 修正将保持不可用。</translation>
    </message>
    <message>
        <source>Stop the setup?</source>
        <translation>停止安装？</translation>
    </message>
    <message>
        <source>The detection stopped responding before any tile came back. Check your connection, then run Detect again (nothing was charged).</source>
        <translation>检测服务已无响应，尚未返回任何瓦片。请检查您的网络连接，然后重新运行检测（未产生任何费用）。</translation>
    </message>
    <message>
        <source>The file may be open in QGIS or in another program. Close it and try Finish again.</source>
        <translation>该文件可能已在 QGIS 或其他程序中打开。请将其关闭后重新点击“完成”。</translation>
    </message>
    <message>
        <source>The imagery is loading slowly... {n}s</source>
        <translation>影像加载较慢... {n}s</translation>
    </message>
    <message>
        <source>The on-device AI could not start, so the AI fix is off. Your detections are safe: switch the fix method to Manual to keep correcting, or save them as they are.</source>
        <translation>本地 AI 无法启动，因此 AI 修正不可用。您的检测结果是安全的：将修正方式切换为手动即可继续修正，或按原样保存。</translation>
    </message>
    <message>
        <source>The on-device AI is unavailable, so the AI fix is off. Switch the fix method to Manual to keep correcting.</source>
        <translation>本地 AI 不可用，因此 AI 修正不可用。请将修正方式切换为手动以继续修正。</translation>
    </message>
    <message>
        <source>The on-device AI will not be installed, so fixing a polygon with it stays unavailable. What is already downloaded is kept, so starting again resumes from there.</source>
        <translation>本地 AI 将不会被安装，因此使用它修正多边形的功能仍不可用。已下载的内容会被保留，因此重新开始时将从该处继续。</translation>
    </message>
    <message>
        <source>The package index refused the download (error 403).

This is usually a company or campus network filtering downloads. Ask your IT administrator to allow pypi.org and files.pythonhosted.org, or run the install from another network.

Automatic (cloud) mode does not need this download.</source>
        <translation>软件包索引拒绝了此次下载（错误 403）。

这通常是公司或校园网络在过滤下载内容。请让您的 IT 管理员允许访问 pypi.org 和 files.pythonhosted.org，或改用其他网络进行安装。

自动（云端）模式不需要此下载。</translation>
    </message>
    <message>
        <source>The raster file could not be found:
{path}

It may have been moved or renamed, or the drive or network share it is on may be disconnected. Reload the layer from where the file is now, then start again.</source>
        <translation>找不到栅格文件：
{path}

该文件可能已被移动或重命名，或其所在的磁盘或网络共享已断开连接。请从文件当前所在位置重新加载图层，然后重新开始。</translation>
    </message>
    <message>
        <source>Thin this outline before you drag its corners, without moving the dials that drive the whole layer.</source>
        <translation>在拖动节点之前先细化此轮廓，不会影响驱动整个图层的调节项。</translation>
    </message>
    <message>
        <source>This area is large for this precision. Raise the precision or zoom in for sharper detections.</source>
        <translation>该区域相对于当前精度偏大。请提高精度或放大地图以获得更精确的检测结果。</translation>
    </message>
    <message>
        <source>This example is very small at this precision. Raise the precision or draw a larger object.</source>
        <translation>此示例在当前精度下非常小。请提高精度，或绘制更大的对象。</translation>
    </message>
    <message>
        <source>This example is very small even at full precision. Draw a larger object, or it may be too small to detect.</source>
        <translation>即使在最高精度下，此示例也非常小。请绘制更大的对象，否则可能太小而无法检测。</translation>
    </message>
    <message>
        <source>This layer has no image over your zone at this precision. The map source answered with an empty tile, so there is nothing to detect on. Lower Precision, zoom the layer out until the imagery shows, or pick a layer that covers this area.</source>
        <translation>在当前精度下，该图层在您的区域内没有影像。地图源返回的是空白瓦片，因此没有可检测的内容。请降低精度、缩小地图直到影像显示，或选择一个覆盖此区域的图层。</translation>
    </message>
    <message>
        <source>This polygon only. Every other one follows the Shapes step.</source>
        <translation>仅此多边形；其他多边形仍遵循“形状”步骤的设置。</translation>
    </message>
    <message>
        <source>This polygon only. Fewer points means fewer corners to drag.</source>
        <translation>仅此多边形；减少节点数即可减少需要拖动的节点。</translation>
    </message>
    <message>
        <source>This run costs more credits than a free run covers.</source>
        <translation>本次运行所需积分超出免费额度。</translation>
    </message>
    <message>
        <source>This zone at this precision needs a subscription. Lower the precision or the zone to stay free.</source>
        <translation>在当前精度下，该区域需要订阅。请降低精度或缩小区域以保持免费。</translation>
    </message>
    <message>
        <source>This zone is too large for sharp detections, even at full precision. Draw a smaller zone for the best results.</source>
        <translation>即使在最高精度下，该区域也过大，难以获得精确的检测结果。请绘制更小的区域以获得最佳效果。</translation>
    </message>
    <message>
        <source>Tiles your layer has no image for are dropped before they are sent, so a run can cost less than this, never more.</source>
        <translation>图层中没有影像的瓦片会在发送前被剔除，因此本次运行消耗的积分可能低于此数，但绝不会更多。</translation>
    </message>
    <message>
        <source>Type a word for the object, or draw an example.</source>
        <translation>为对象输入一个词，或绘制一个示例。</translation>
    </message>
    <message>
        <source>Upgrade to Pro to finish this zone: 10,000 credits/month.</source>
        <translation>升级到 Pro 以完成该区域的检测：每月 10,000 积分。</translation>
    </message>
    <message>
        <source>Wait for the on-device AI to finish installing.</source>
        <translation>等待本地 AI 完成安装。</translation>
    </message>
    <message>
        <source>{count} polygon(s) exported</source>
        <translation>已 Export {count} 个多边形</translation>
    </message>
    <message>
        <source>{n} {object} saved</source>
        <translation>已保存 {n} 个{object}</translation>
    </message>
    <message>
        <source>{n} {object} saved to {layer}</source>
        <translation>已将 {n} 个{object}保存到 {layer}</translation>
    </message>
    <message>
        <source>{used} credits</source>
        <translation>{used} 积分</translation>
    </message>
    <!-- Correct-step on-device AI setup, free-run cap wording, missing raster (2026-07-30) -->
    <message>
        <source>or lower the detail to stay free.</source>
        <translation>或降低细节以保持免费。</translation>
    </message>
    <message>
        <source>This zone at this detail needs a subscription.</source>
        <translation>在此细节下处理该区域需要订阅。</translation>
    </message>
    <message>
        <source>Subscribe</source>
        <translation>订阅</translation>
    </message>
</context>
</TS>
