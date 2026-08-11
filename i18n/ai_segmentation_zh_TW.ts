<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE TS>
<TS version="2.1" language="zh_TW">
<context>
    <name>AISegmentation</name>
    <message>
        <source>Normal</source>
        <translation>正常</translation>
    </message>
    <message>
        <source>Confidence</source>
        <translation>信心度</translation>
    </message>
    <message>
        <source>Outline</source>
        <translation>輪廓</translation>
    </message>
    <message>
        <source>Retry</source>
        <translation>重試</translation>
    </message>
    <message>
        <source>Click to enlarge</source>
        <translation>點擊放大</translation>
    </message>
    <message>
        <source>This is exactly what the AI uses: your object with a little of its surroundings.</source>
        <translation>這正是 AI 所使用的內容：您的物件加上周圍一小部分環境。</translation>
    </message>
    <message>
        <source>Describe what to find</source>
        <translation>描述要尋找的內容</translation>
    </message>
    <message>
        <source>Exclude a look-alike</source>
        <translation>排除相似物件</translation>
    </message>
    <message>
        <source>Too generic to name. Clear the box to search from your example alone, or type a concrete object.</source>
        <translation>名稱太籠統。清空欄位僅以範例搜尋，或輸入具體的物件名稱。</translation>
    </message>
    <message>
        <source>Mark an object to find more like it.</source>
        <translation>標記一個物件以尋找更多類似的物件。</translation>
    </message>
    <message>
        <source>Mark a false positive to drop things like it.</source>
        <translation>標記一個誤判物件，以剔除類似的結果。</translation>
    </message>
    <message>
        <source>Finish or exit the review to switch modes.</source>
        <translation>請先完成或退出檢視，才能切換模式。</translation>
    </message>
    <message>
        <source>Detection</source>
        <translation>偵測結果</translation>
    </message>
    <message>
        <source>Confidence:</source>
        <translation>信心度：</translation>
    </message>
    <message>
        <source>Minimum confidence to keep a detected object. Lower finds more objects but may add false positives; raise it for cleaner results on large, distinct features.</source>
        <translation>保留偵測物件所需的最低信心度。數值越低可偵測到更多物件，但可能增加誤判；數值越高則能在大型、明顯的地物上取得更乾淨的結果。</translation>
    </message>
    <message>
        <source>Dependencies</source>
        <translation>相依套件</translation>
    </message>
    <message>
        <source>Local AI model files stored on this computer.</source>
        <translation>儲存在這台電腦上的本機 AI 模型檔案。</translation>
    </message>
    <message>
        <source>On disk</source>
        <translation>佔用空間</translation>
    </message>
    <message>
        <source>Not installed</source>
        <translation>未安裝</translation>
    </message>
    <message>
        <source>Open folder</source>
        <translation>開啟資料夾</translation>
    </message>
    <message>
        <source>Accept the Terms and Privacy Policy to enable segmentation.</source>
        <translation>請接受服務條款與隱私政策以啟用分割功能。</translation>
    </message>
    <message>
        <source>I agree to the &lt;a href=&quot;{terms}&quot;&gt;Terms&lt;/a&gt; and &lt;a href=&quot;{privacy}&quot;&gt;Privacy Policy&lt;/a&gt;</source>
        <translation>我同意&lt;a href=&quot;{terms}&quot;&gt;服務條款&lt;/a&gt;與&lt;a href=&quot;{privacy}&quot;&gt;隱私政策&lt;/a&gt;</translation>
    </message>
    <message>
        <source>No valid polygons could be created from the selection. Try adjusting the refine settings or making a new selection.</source>
        <translation>無法從此選取範圍建立有效的多邊形。請試著調整細修設定或重新選取。</translation>
    </message>
    <message>
        <source>Repairing Installation</source>
        <translation>正在修復安裝</translation>
    </message>
    <message>
        <source>Repairing installation...</source>
        <translation>正在修復安裝...</translation>
    </message>
    <message>
        <source>Restart QGIS Required</source>
        <translation>需要重新啟動 QGIS</translation>
    </message>
    <message>
        <source>Something went wrong with this click, so it was not applied. Please try again.</source>
        <translation>此次點擊發生問題，未套用變更，請再試一次。</translation>
    </message>
    <message>
        <source>The Python runtime used by the AI engine is damaged (this can be caused by a disk cleanup tool or antivirus). It will now be repaired automatically. Please try your selection again once the repair finishes.</source>
        <translation>AI 引擎所使用的 Python 執行環境已損毀（可能是磁碟清理工具或防毒軟體所造成）。系統現在會自動修復，修復完成後請重新選取一次。</translation>
    </message>
    <message>
        <source>Your polygons were added as a temporary layer so nothing is lost.</source>
        <translation>您的多邊形已加入暫時圖層，不會遺失任何內容。</translation>
    </message>
    <message>
        <source>Could not write to {name}. Saved to a separate file instead.</source>
        <translation>無法寫入 {name}，已改存為另一個檔案。</translation>
    </message>
    <message>
        <source>Click Install to set up AI Segmentation</source>
        <translation>點擊「安裝」以設定 AI Segmentation</translation>
    </message>
    <message>
        <source>Installing AI Segmentation...</source>
        <translation>正在安裝 AI Segmentation...</translation>
    </message>
    <message>
        <source>Verifying installation...</source>
        <translation>正在驗證安裝...</translation>
    </message>
    <message>
        <source>Detecting device...</source>
        <translation>正在偵測裝置...</translation>
    </message>
    <message>
        <source>Install path: {}</source>
        <translation>安裝路徑：{}</translation>
    </message>
    <message>
        <source>To install in a different folder, set the environment variable AI_SEGMENTATION_CACHE_DIR:</source>
        <translation>若要安裝到其他資料夾，請設定環境變數 AI_SEGMENTATION_CACHE_DIR：</translation>
    </message>
    <message>
        <source>1. Open Windows Settings &gt; System &gt; Advanced system settings
2. Click 'Environment Variables'
3. Under 'User variables', click 'New'
4. Variable name: AI_SEGMENTATION_CACHE_DIR
5. Variable value: the folder path you want to use
6. Click OK and restart QGIS</source>
        <translation>1. 開啟 Windows 設定 &gt; 系統 &gt; 進階系統設定
2. 點擊「環境變數」
3. 在「使用者變數」下，點擊「新增」
4. 變數名稱：AI_SEGMENTATION_CACHE_DIR
5. 變數值：您想使用的資料夾路徑
6. 點擊確定並重新啟動 QGIS</translation>
    </message>
    <message>
        <source>Run this command in Terminal, then restart QGIS:

launchctl setenv AI_SEGMENTATION_CACHE_DIR /your/path</source>
        <translation>請在終端機執行以下指令，然後重新啟動 QGIS：

launchctl setenv AI_SEGMENTATION_CACHE_DIR /your/path</translation>
    </message>
    <message>
        <source>Add this line to your ~/.bashrc or ~/.profile, then restart QGIS:

export AI_SEGMENTATION_CACHE_DIR=/your/path</source>
        <translation>請將以下這行加入您的 ~/.bashrc 或 ~/.profile，然後重新啟動 QGIS：

export AI_SEGMENTATION_CACHE_DIR=/your/path</translation>
    </message>
    <message>
        <source>Checking...</source>
        <translation>正在檢查...</translation>
    </message>
    <message>
        <source>Install</source>
        <translation>安裝</translation>
    </message>
    <message>
        <source>Update</source>
        <translation>更新</translation>
    </message>
    <message>
        <source>Downloading AI model...</source>
        <translation>正在下載 AI 模型...</translation>
    </message>
    <message>
        <source>Cancel</source>
        <translation>取消</translation>
    </message>
    <message>
        <source>Cancel installation</source>
        <translation>取消安裝</translation>
    </message>
    <message>
        <source>Are you sure you want to cancel the installation?</source>
        <translation>您確定要取消安裝嗎？</translation>
    </message>
    <message>
        <source>Installation cancelled</source>
        <translation>安裝已取消</translation>
    </message>
    <message>
        <source>Installation failed</source>
        <translation>安裝失敗</translation>
    </message>
    <message>
        <source>Verification failed:</source>
        <translation>驗證失敗：</translation>
    </message>
    <message>
        <source>Verification Failed</source>
        <translation>驗證失敗</translation>
    </message>
    <message>
        <source>Unknown error</source>
        <translation>未知錯誤</translation>
    </message>
    <message>
        <source>Installation Failed</source>
        <translation>安裝失敗</translation>
    </message>
    <message>
        <source>Update QGIS to 3.34+ for the latest AI model</source>
        <translation>將 QGIS 更新至 3.34 以上版本以使用最新的 AI 模型</translation>
    </message>
    <message>
        <source>Download Failed</source>
        <translation>下載失敗</translation>
    </message>
    <message>
        <source>Failed to download model:</source>
        <translation>模型下載失敗：</translation>
    </message>
    <message>
        <source>AI Segmentation by TerraLab</source>
        <translation>AI Segmentation by TerraLab</translation>
    </message>
    <message>
        <source>Select a raster layer (GeoTIFF, WMS, XYZ tiles, etc.)</source>
        <translation>選擇點陣圖圖層（GeoTIFF、WMS、XYZ 圖磚等）</translation>
    </message>
    <message>
        <source>Save polygon</source>
        <translation>儲存多邊形</translation>
    </message>
    <message>
        <source>Undo last point</source>
        <translation>復原上一個點</translation>
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
        <translation>空格鍵</translation>
    </message>
    <message>
        <source>Hold and move to pan the map</source>
        <translation>按住並移動以平移地圖</translation>
    </message>
    <message>
        <source>The AI model works best on one element at a time.</source>
        <translation>AI 模型在一次處理一個地物時效果最佳。</translation>
    </message>
    <message>
        <source>Save your polygon before selecting the next element.</source>
        <translation>選取下一個地物之前，請先儲存您的多邊形。</translation>
    </message>
    <message>
        <source>Export polygon to a layer</source>
        <translation>Export 多邊形至圖層</translation>
    </message>
    <message>
        <source>Export {count} polygons to a layer</source>
        <translation>Export {count} 個多邊形至圖層</translation>
    </message>
    <message>
        <source>Refine selection</source>
        <translation>細修選取範圍</translation>
    </message>
    <message>
        <source>Fill interior holes in the selection</source>
        <translation>填補選取範圍內部的孔洞</translation>
    </message>
    <message>
        <source>Shape</source>
        <translation>形狀</translation>
    </message>
    <message>
        <source>Size</source>
        <translation>大小</translation>
    </message>
    <message>
        <source>Outline</source>
        <translation>輪廓</translation>
    </message>
    <message>
        <source>Left-click to select</source>
        <translation>左鍵點擊以選取</translation>
    </message>
    <message>
        <source>Left-click to add more</source>
        <translation>左鍵點擊以新增更多</translation>
    </message>
    <message>
        <source>Right-click to exclude from selection</source>
        <translation>右鍵點擊以從選取範圍中排除</translation>
    </message>
    <message>
        <source>Invalid Layer</source>
        <translation>無效的圖層</translation>
    </message>
    <message>
        <source>Not Ready</source>
        <translation>尚未就緒</translation>
    </message>
    <message>
        <source>Layer Creation Failed</source>
        <translation>建立圖層失敗</translation>
    </message>
    <message>
        <source>Could not create the output layer.</source>
        <translation>無法建立輸出圖層。</translation>
    </message>
    <message>
        <source>Export Failed</source>
        <translation>Export 失敗</translation>
    </message>
    <message>
        <source>Could not save layer to file:</source>
        <translation>無法將圖層儲存至檔案：</translation>
    </message>
    <message>
        <source>You have {count} unsaved polygon(s).</source>
        <translation>您有 {count} 個尚未儲存的多邊形。</translation>
    </message>
    <message>
        <source>Changing layer will discard your current segmentation. Continue?</source>
        <translation>變更圖層將捨棄目前的分割結果，是否繼續？</translation>
    </message>
    <message>
        <source>Change Layer?</source>
        <translation>變更圖層？</translation>
    </message>
    <message>
        <source>Stop Segmentation?</source>
        <translation>停止分割？</translation>
    </message>
    <message>
        <source>This will discard {count} polygon(s).</source>
        <translation>這將捨棄 {count} 個多邊形。</translation>
    </message>
    <message>
        <source>Use 'Export to layer' to keep them.</source>
        <translation>請使用「Export 至圖層」來保留它們。</translation>
    </message>
    <message>
        <source>Do you want to continue?</source>
        <translation>是否要繼續？</translation>
    </message>
    <message>
        <source>Edit saved polygon</source>
        <translation>編輯已儲存的多邊形</translation>
    </message>
    <message>
        <source>Warning: you are about to edit an already saved polygon.</source>
        <translation>警告：您即將編輯一個已儲存的多邊形。</translation>
    </message>
    <message>
        <source>Contact us</source>
        <translation>聯絡我們</translation>
    </message>
    <message>
        <source>Bug, question, feature request?</source>
        <translation>錯誤回報、問題或功能建議？</translation>
    </message>
    <message>
        <source>Copy email address</source>
        <translation>複製電子郵件地址</translation>
    </message>
    <message>
        <source>or</source>
        <translation>或</translation>
    </message>
    <message>
        <source>Book a video call</source>
        <translation>預約視訊通話</translation>
    </message>
    <message>
        <source>Tutorial</source>
        <translation>教學</translation>
    </message>
    <message>
        <source>Settings</source>
        <translation>設定</translation>
    </message>
    <message>
        <source>Terms</source>
        <translation>條款</translation>
    </message>
    <message>
        <source>Privacy</source>
        <translation>隱私權</translation>
    </message>
    <message>
        <source>Segment elements on raster images using AI</source>
        <translation>使用 AI 分割點陣影像中的地物</translation>
    </message>
    <message>
        <source>Copy your logs with the button below and send them to our support email.</source>
        <translation>使用下方按鈕複製您的記錄檔，並寄至我們的支援信箱。</translation>
    </message>
    <message>
        <source>We'll get this fixed for you :)</source>
        <translation>我們會盡快為您修復 :)</translation>
    </message>
    <message>
        <source>1. Click to copy logs</source>
        <translation>1. 點擊以複製日誌</translation>
    </message>
    <message>
        <source>2. Click to send to {}</source>
        <translation>2. 點擊以寄送至 {}</translation>
    </message>
    <message>
        <source>Open email client</source>
        <translation>開啟電子郵件用戶端</translation>
    </message>
    <message>
        <source>Copied!</source>
        <translation>已複製！</translation>
    </message>
    <message>
        <source>SSL Certificate Error</source>
        <translation>SSL 憑證錯誤</translation>
    </message>
    <message>
        <source>Installation Blocked</source>
        <translation>安裝被封鎖</translation>
    </message>
    <message>
        <source>Click is outside the &apos;{layer}&apos; raster. To segment another raster, stop the current segmentation first.</source>
        <translation>點擊位置在「{layer}」點陣圖範圍之外。若要分割其他點陣圖，請先停止目前的分割。</translation>
    </message>
    <message>
        <source>{ext} format is not directly supported. GDAL is not available.
Please convert your raster to GeoTIFF (.tif) before using AI Segmentation.</source>
        <translation>不直接支援 {ext} 格式，GDAL 無法使用。
請先將您的點陣圖轉換為 GeoTIFF（.tif）格式，再使用 AI Segmentation。</translation>
    </message>
    <message>
        <source>Cannot open {ext} file. The format may not be supported by your QGIS installation.
Please convert your raster to GeoTIFF (.tif) before using AI Segmentation.</source>
        <translation>無法開啟 {ext} 檔案，您的 QGIS 安裝可能不支援此格式。
請先將您的點陣圖轉換為 GeoTIFF（.tif）格式，再使用 AI Segmentation。</translation>
    </message>
    <message>
        <source>Failed to read {ext} file: {error}
Please convert your raster to GeoTIFF (.tif) manually.</source>
        <translation>讀取 {ext} 檔案失敗：{error}
請手動將您的點陣圖轉換為 GeoTIFF（.tif）格式。</translation>
    </message>
    <message>
        <source>PyTorch cannot load on Windows</source>
        <translation>PyTorch 在 Windows 上無法載入</translation>
    </message>
    <message>
        <source>The plugin requires Visual C++ Redistributables to run PyTorch.

Please download and install:
https://aka.ms/vs/17/release/vc_redist.x64.exe

After installation, restart QGIS and try again.</source>
        <translation>此外掛程式需要安裝 Visual C++ Redistributables 才能執行 PyTorch。

請下載並安裝：
https://aka.ms/vs/17/release/vc_redist.x64.exe

安裝完成後，請重新啟動 QGIS 並再試一次。</translation>
    </message>
    <message>
        <source>Segmentation failed</source>
        <translation>分割失敗</translation>
    </message>
    <message>
        <source>Layer data provider is not available.</source>
        <translation>圖層資料提供者無法使用。</translation>
    </message>
    <message>
        <source>Failed to fetch tiles from the online layer. Check your network connection.</source>
        <translation>無法從線上圖層取得圖磚，請檢查您的網路連線。</translation>
    </message>
    <message>
        <source>Crop Error</source>
        <translation>裁切錯誤</translation>
    </message>
    <message>
        <source>No raster file path available. Please restart segmentation.</source>
        <translation>沒有可用的點陣圖檔案路徑，請重新啟動分割。</translation>
    </message>
    <message>
        <source>Encoding Error</source>
        <translation>編碼錯誤</translation>
    </message>
    <message>
        <source>Disconnected parts detected. For best accuracy, segment one element at a time.</source>
        <translation>偵測到不連續的部分。為求最佳準確度，請一次分割一個地物。</translation>
    </message>
    <message>
        <source>Updating...</source>
        <translation>正在更新...</translation>
    </message>
    <message>
        <source>Missing Visual C++ Redistributable. Install it, restart your computer, then click Retry.</source>
        <translation>缺少 Visual C++ Redistributable。請安裝後重新啟動電腦，再點擊「重試」。</translation>
    </message>
    <message>
        <source>Retry</source>
        <translation>重試</translation>
    </message>
    <message>
        <source>Loading AI model...</source>
        <translation>正在載入 AI 模型...</translation>
    </message>
    <message>
        <source>Ready</source>
        <translation>就緒</translation>
    </message>
    <message>
        <source>Model load failed</source>
        <translation>模型載入失敗</translation>
    </message>
    <message>
        <source>New here?</source>
        <translation>第一次使用嗎？</translation>
    </message>
    <message>
        <source>Watch the tutorial</source>
        <translation>觀看教學影片</translation>
    </message>
    <message>
        <source>Network Connection Problem</source>
        <translation>網路連線問題</translation>
    </message>
    <message>
        <source>Your connection appears unstable or blocked. Check: (1) your internet is working, (2) QGIS > Settings > Options > Network has a proxy configured if you are on a corporate network, (3) your firewall allows connections to pypi.org and files.pythonhosted.org.</source>
        <translation>您的連線似乎不穩定或已被封鎖。請檢查：(1) 網路是否正常運作，(2) 若您在公司網路環境中，QGIS > 偏好設定 > 選項 > 網路是否已設定代理伺服器，(3) 防火牆是否允許連線至 pypi.org 及 files.pythonhosted.org。</translation>
    </message>
    <message>
        <source>Checking...</source>
        <translation>正在檢查...</translation>
    </message>
    <message>
        <source>AI Edit</source>
        <translation>AI Edit</translation>
    </message>
    <message>
        <source>Generate imagery with AI on map zones (opens AI Edit plugin)</source>
        <translation>在地圖範圍上使用 AI 生成影像（開啟 AI Edit 外掛程式）</translation>
    </message>
    <message>
        <source>Right-click must be inside the current selection area.</source>
        <translation>右鍵點擊必須在目前的選取範圍內。</translation>
    </message>
    <message>
        <source>Account Settings</source>
        <translation>帳戶設定</translation>
    </message>
    <message>
        <source>Loading account info...</source>
        <translation>正在載入帳戶資訊...</translation>
    </message>
    <message>
        <source>Model File Corrupted</source>
        <translation>模型檔案損毀</translation>
    </message>
    <message>
        <source>Re-downloading Model</source>
        <translation>正在重新下載模型</translation>
    </message>
    <message>
        <source>The AI model file was corrupted and is being re-downloaded. Please try your selection again once it finishes.</source>
        <translation>AI 模型檔案已損毀，正在重新下載。下載完成後請重新選取一次。</translation>
    </message>
    <message>
        <source>The AI model file is corrupted but could not be removed automatically. Please delete this folder and restart QGIS:</source>
        <translation>AI 模型檔案已損毀，但無法自動移除。請刪除此資料夾並重新啟動 QGIS：</translation>
    </message>
    <message>
        <source>Segment your map with AI</source>
        <translation>使用 AI 分割您的地圖</translation>
    </message>
    <message>
        <source>Sign in / Sign up to start</source>
        <translation>登入／註冊以開始使用</translation>
    </message>
    <message>
        <source>Sign in via your browser to start using AI Segmentation</source>
        <translation>透過瀏覽器登入以開始使用 AI Segmentation</translation>
    </message>
    <message>
        <source>Open again</source>
        <translation>再次開啟</translation>
    </message>
    <message>
        <source>Didn't open? Open the page again</source>
        <translation>沒有開啟嗎？再次開啟頁面</translation>
    </message>
    <message>
        <source>Sign-in timed out. Click Connect to try again.</source>
        <translation>登入逾時。請點擊「連線」再試一次。</translation>
    </message>
    <message>
        <source>Sign-in was cancelled in the browser. Click Connect to try again.</source>
        <translation>已在瀏覽器中取消登入。請點擊「連線」再試一次。</translation>
    </message>
    <message>
        <source>Unexpected response from the server. Please try again.</source>
        <translation>伺服器傳回未預期的回應，請再試一次。</translation>
    </message>
    <message>
        <source>This account has no active AI Segmentation plan. Reactivate it on terra-lab.ai, then click Connect again.</source>
        <translation>此帳戶沒有生效中的 AI Segmentation 方案。請在 terra-lab.ai 重新啟用，再點擊「連線」。</translation>
    </message>
    <message>
        <source>Connecting AI Segmentation</source>
        <translation>正在連線 AI Segmentation</translation>
    </message>
    <message>
        <source>Cancelling sign-in</source>
        <translation>正在取消登入</translation>
    </message>
    <message>
        <source>Help / Report a problem</source>
        <translation>說明／回報問題</translation>
    </message>
    <message>
        <source>Report a problem</source>
        <translation>回報問題</translation>
    </message>
    <message>
        <source>Connected</source>
        <translation>已連線</translation>
    </message>
    <message>
        <source>Sign out</source>
        <translation>登出</translation>
    </message>
    <message>
        <source>Sign out of AI Segmentation?</source>
        <translation>要登出 AI Segmentation 嗎？</translation>
    </message>
    <message>
        <source>You can sign back in anytime from QGIS.</source>
        <translation>您可以隨時從 QGIS 重新登入。</translation>
    </message>
    <message>
        <source>Active</source>
        <translation>生效中</translation>
    </message>
    <message>
        <source>Make this map presentation-ready</source>
        <translation>讓這份地圖立即可用於簡報</translation>
    </message>
    <message>
        <source>AI Edit: turn your imagery into presentation and planning visuals</source>
        <translation>AI Edit：將您的影像轉換為簡報與規劃視覺圖</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Mode selection</source>
        <translation>模式選擇</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Stop the active segmentation before switching modes.</source>
        <translation>請先停止目前進行中的分割，才能切換模式。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Cancel the active detection before switching modes.</source>
        <translation>請先取消目前進行中的偵測，才能切換模式。</translation>
    </message>
    <message>
        <location filename="../src/ui/zone_selection_maptool.py" line="0"/>
        <source>Cancel the running detection first</source>
        <translation>請先取消進行中的偵測</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Zone too large - reduce the selection area</source>
        <translation>範圍過大－請縮小選取範圍</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Detect objects</source>
        <translation>偵測物件</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Cancel detection</source>
        <translation>取消偵測</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Tile {current}/{total}</source>
        <translation>圖磚 {current}/{total}</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Spot reserved · starting in ~{eta}</source>
        <translation>已保留名額·約 {eta} 後開始</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>{s} seconds</source>
        <translation>{s} 秒</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>{m} min</source>
        <translation>{m} 分鐘</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>{n} credits remaining</source>
        <translation>剩餘 {n} 點數</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>{n} free detection(s) remaining</source>
        <translation>剩餘 {n} 次免費偵測</translation>
    </message>
    <message>
        <location filename="../src/ui/account_settings_dialog.py" line="0"/>
        <source>{remaining} / {total} credits</source>
        <translation>{remaining} / {total} 點數</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Auto detection (live)</source>
        <translation>自動偵測（即時）</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Preparing tiles...</source>
        <translation>正在準備圖磚...</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Cancelling...</source>
        <translation>正在取消...</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Finishing the previous run, please wait a moment...</source>
        <translation>正在完成上一次的執行，請稍候...</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Detection failed. Check your connection and try again.</source>
        <translation>偵測失敗，請檢查您的連線並再試一次。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Automatic detection is temporarily unavailable. Please try again later.</source>
        <translation>自動偵測暫時無法使用，請稍後再試。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Draw a zone first. Automatic detection on online layers needs a zone.</source>
        <translation>請先繪製範圍。線上圖層的自動偵測需要指定範圍。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>The zone is outside the selected raster layer. Pick the right layer or redraw the zone.</source>
        <translation>範圍超出所選點陣圖圖層之外。請選擇正確的圖層或重新繪製範圍。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Less</source>
        <translation>較少</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>More</source>
        <translation>較多</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Available when detection finishes</source>
        <translation>偵測完成後即可使用</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Off</source>
        <translation>關閉</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>No limit</source>
        <translation>無限制</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Hide detections smaller than this ground area. Use it to drop tiny noise blobs. 0 = keep all.</source>
        <translation>隱藏面積小於此地面面積的偵測結果，可用於剔除微小的雜訊斑點。0＝全部保留。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Hide detections larger than this ground area. 0 = no limit.</source>
        <translation>隱藏面積大於此地面面積的偵測結果。0＝無限制。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Saved {n} polygon(s) to {name}</source>
        <translation>已將 {n} 個多邊形儲存至 {name}</translation>
    </message>
    <message>
        <location filename="../src/api/terralab_client.py" line="0"/>
        <source>Cannot reach the server. Check your internet connection.</source>
        <translation>無法連上伺服器，請檢查您的網路連線。</translation>
    </message>
    <message>
        <location filename="../src/api/terralab_client.py" line="0"/>
        <source>Server refused the connection.</source>
        <translation>伺服器拒絕連線。</translation>
    </message>
    <message>
        <location filename="../src/api/terralab_client.py" line="0"/>
        <source>Request timed out. Check your connection or try again.</source>
        <translation>請求逾時，請檢查您的連線或再試一次。</translation>
    </message>
    <message>
        <location filename="../src/api/terralab_client.py" line="0"/>
        <source>SSL certificate error. Your network may be blocking secure connections.</source>
        <translation>SSL 憑證錯誤，您的網路可能封鎖了安全連線。</translation>
    </message>
    <message>
        <location filename="../src/api/terralab_client.py" line="0"/>
        <source>Proxy connection failed. Check QGIS proxy settings (Settings &gt; Options &gt; Network).</source>
        <translation>代理伺服器連線失敗。請檢查 QGIS 代理伺服器設定（偏好設定 &gt; 選項 &gt; 網路）。</translation>
    </message>
    <message>
        <location filename="../src/api/terralab_client.py" line="0"/>
        <source>Authentication failed. Please sign in again.</source>
        <translation>驗證失敗，請重新登入。</translation>
    </message>
    <message>
        <location filename="../src/api/terralab_client.py" line="0"/>
        <source>Network error. Check your internet connection.</source>
        <translation>網路錯誤，請檢查您的網路連線。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Checking your AI Segmentation subscription</source>
        <translation>正在檢查您的 AI Segmentation 訂閱</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Loading AI Segmentation settings</source>
        <translation>正在載入 AI Segmentation 設定</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Refreshing credits</source>
        <translation>正在重新整理點數</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Warming up AI Segmentation</source>
        <translation>正在準備 AI Segmentation</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Popular</source>
        <translation>熱門</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Library</source>
        <translation>範例庫</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Use just 1-2 words for the object.</source>
        <translation>物件名稱請只用 1-2 個字。</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Type the object itself, not a sentence or question.</source>
        <translation>請直接輸入物件名稱，而非句子或問句。</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Too generic. Draw an example instead, or use a concrete word like building.</source>
        <translation>太籠統了。請改為繪製範例，或使用具體的字詞，例如 building。</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Name a concrete object, not how it looks.</source>
        <translation>請命名具體的物件，而非描述其外觀。</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Segment one object - drop words like 'near' or 'with'.</source>
        <translation>請分割單一物件－請去除像「near」或「with」之類的字詞。</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Use a real object word.</source>
        <translation>請使用真實存在的物件名稱。</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Use a 1-2 word object name.</source>
        <translation>請使用 1-2 個字的物件名稱。</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Did you mean '{term}'?</source>
        <translation>您是指「{term}」嗎？</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Loading...</source>
        <translation>正在載入...</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>No preview</source>
        <translation>無預覽</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>No preview yet</source>
        <translation>尚無預覽</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Segment library</source>
        <translation>分割範例庫</translation>
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
        <translation>{n} 次偵測</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>{n} object(s)</source>
        <translation>{n} 個物件</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Loading segment library</source>
        <translation>正在載入分割範例庫</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Search objects... e.g. building, solar panel</source>
        <translation>搜尋物件... 例如：building、solar panel</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Fuzzy edges: this one may need cleanup after detection.</source>
        <translation>邊緣模糊：此物件偵測後可能需要清理。</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Use this prompt</source>
        <translation>使用此提示詞</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library_dialog.py" line="0"/>
        <source>Use</source>
        <translation>使用</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>≈ {n} tiles = {n} credits</source>
        <translation>≈ {n} 個圖磚＝{n} 點數</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_plugin.py" line="0"/>
        <source>Draw your example inside the selected zone.</source>
        <translation>在所選範圍內繪製您的範例。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>{n} objects found</source>
        <translation>找到 {n} 個物件</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>No objects found</source>
        <translation>未找到任何物件</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>0 shown at {pct}% - lower Confidence to reveal them</source>
        <translation>在 {pct}% 下顯示 0 個－請降低信心度以顯示它們</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>More objects</source>
        <translation>更多物件</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Only confident</source>
        <translation>僅顯示確定的</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Export {n} polygons</source>
        <translation>Export {n} 個多邊形</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Lower Confidence to show objects first.</source>
        <translation>請先降低信心度以顯示物件。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Discard these detections?</source>
        <translation>要捨棄這些偵測結果嗎？</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Your {total} detections will be discarded. You keep your zone, object and settings. Running Detect again will use new credits.</source>
        <translation>您的 {total} 個偵測結果將被捨棄。範圍、物件與設定會保留。重新執行偵測會耗費新的點數。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Discard &amp;&amp; adjust</source>
        <translation>捨棄並調整</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Keep your detections?</source>
        <translation>要保留您的偵測結果嗎？</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Save &amp;&amp; exit</source>
        <translation>儲存並離開</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Discard &amp;&amp; exit</source>
        <translation>捨棄並離開</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>How many objects sit at each confidence level.</source>
        <translation>各信心度層級所包含的物件數量。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>Every building, tree, or road as clean polygons</source>
        <translation>每一棟建築物、每一棵樹或每一條道路都會以乾淨的多邊形呈現</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>Cancel anytime; your exported layers stay yours</source>
        <translation>隨時可取消；已 Export 的圖層永遠屬於您</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>Opens your TerraLab dashboard</source>
        <translation>開啟您的 TerraLab 控制台</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>Start Automatic AI Segmentation</source>
        <translation>開始自動 AI Segmentation</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>Draw your zone</source>
        <translation>繪製您的範圍</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>Click on the map to outline the area to scan.</source>
        <translation>點擊地圖以圈選要掃描的區域。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>Keep clicking around the area, at least 3 points.</source>
        <translation>持續點擊以圈選區域，至少需要 3 個點。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>Click the first point to close the zone.</source>
        <translation>點擊起點以封閉範圍。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>undo point</source>
        <translation>復原點</translation>
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
        <translation>瀏覽附有前後對比預覽的現成物件。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>Filter detections by confidence. Lower shows more (and noisier), higher keeps only the strongest. Free and instant.</source>
        <translation>依信心度篩選偵測結果。數值越低顯示越多（但雜訊也越多），數值越高則只保留最可靠的結果。免費且即時。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_build.py" line="0"/>
        <source>Show tiles (debug)</source>
        <translation>顯示圖磚（偵錯）</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>"{word}" will run as "{token}".</source>
        <translation>「{word}」將以「{token}」執行。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>That word isn't recognized - try a common object like building or tree.</source>
        <translation>無法辨識該字詞－請嘗試常見的物件，例如 building 或 tree。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>One object per run - start with the first one, then run again.</source>
        <translation>每次執行僅限一種物件－請先處理第一種，再重新執行。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>The Library has ready-to-use objects.</source>
        <translation>範例庫中有現成的物件可供使用。</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/cards.py" line="0"/>
        <source>Older detection</source>
        <translation>較早的偵測結果</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Details</source>
        <translation>詳細資訊</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Fullscreen</source>
        <translation>全螢幕</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Exit fullscreen</source>
        <translation>退出全螢幕</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Prompt</source>
        <translation>提示詞</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Copy</source>
        <translation>複製</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Copy prompt</source>
        <translation>複製提示詞</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Copied</source>
        <translation>已複製</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Template</source>
        <translation>範本</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Your detection</source>
        <translation>您的偵測結果</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Open the Library from the Automatic page to use this.</source>
        <translation>請從自動頁面開啟範例庫以使用此功能。</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>DATE</source>
        <translation>日期</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>OBJECTS</source>
        <translation>物件</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>CREDITS</source>
        <translation>點數</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>TILES</source>
        <translation>圖磚</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>RESOLUTION</source>
        <translation>解析度</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>EXAMPLE</source>
        <translation>範例</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Used</source>
        <translation>已使用</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Restore to map</source>
        <translation>還原至地圖</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Reopens this run's review at the same place. Free - no credits.</source>
        <translation>在原處重新開啟此次執行的檢視畫面。免費－不耗費點數。</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Export...</source>
        <translation>Export...</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Remove from favorites</source>
        <translation>從收藏中移除</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Add to favorites</source>
        <translation>加入收藏</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Format:</source>
        <translation>格式：</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>GeoPackage keeps the embedded style; other formats are saved without a style.</source>
        <translation>GeoPackage 會保留內嵌的樣式；其他格式則不含樣式儲存。</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/detail.py" line="0"/>
        <source>Browse...</source>
        <translation>瀏覽...</translation>
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
        <source>Load older runs</source>
        <translation>載入較早的執行紀錄</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/dialog.py" line="0"/>
        <source>Nothing here yet. Your automatic detections will land here, ready to reuse, restore or export.</source>
        <translation>目前尚無內容。您的自動偵測結果會顯示於此，可重複使用、還原或 Export。</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/dialog.py" line="0"/>
        <source>Could not load this run's stored detections. Try again later.</source>
        <translation>無法載入此次執行儲存的偵測結果，請稍後再試。</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/dialog.py" line="0"/>
        <source>Nothing to export at this confidence. Lower it and try again.</source>
        <translation>在此信心度下沒有可 Export 的內容，請降低後再試一次。</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/dialog.py" line="0"/>
        <source>The export failed. Check the file path and try again.</source>
        <translation>Export 失敗，請檢查檔案路徑並再試一次。</translation>
    </message>
    <message>
        <location filename="../src/ui/dialogs/segment_library/dialog.py" line="0"/>
        <source>Exported {n} polygon(s).</source>
        <translation>已 Export {n} 個多邊形。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>Add a point</source>
        <translation>新增一個點</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>Add area</source>
        <translation>新增區域</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>Arrow keys</source>
        <translation>方向鍵</translation>
    </message>
    <message>
        <location filename="../src/api/terralab_client.py" line="0"/>
        <source>Automatic</source>
        <translation>自動</translation>
    </message>
    <message>
        <location filename="../src/core/feature_encoder.py" line="0"/>
        <source>Click</source>
        <translation>點擊</translation>
    </message>
    <message>
        <location filename="../src/ui/plugin/auto_run.py" line="0"/>
        <source>Could not place the example on the image. Redraw the example box inside the zone and try again.</source>
        <translation>無法將範例放置於影像上。請在範圍內重新繪製範例方框並再試一次。</translation>
    </message>
    <message>
        <location filename="../src/core/run_restore.py" line="0"/>
        <source>Could not rebuild this run's detections.</source>
        <translation>無法重建此次執行的偵測結果。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>Delete the active object</source>
        <translation>刪除目前的物件</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Detection continues in the background. Reopen AI Segmentation to follow it.</source>
        <translation>偵測將在背景中繼續進行。重新開啟 AI Segmentation 即可追蹤進度。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>Double-click</source>
        <translation>雙擊</translation>
    </message>
    <message>
        <location filename="../src/core/pip_diagnostics.py" line="0"/>
        <source>Example</source>
        <translation>範例</translation>
    </message>
    <message>
        <location filename="../src/core/run_restore.py" line="0"/>
        <source>Finish or exit the current run before restoring a past one.</source>
        <translation>請先完成或退出目前的執行，才能還原過去的執行紀錄。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>Finish the zone</source>
        <translation>完成範圍</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>General</source>
        <translation>一般</translation>
    </message>
    <message>
        <location filename="../src/ui/plugin/manual_handoff.py" line="0"/>
        <source>Install now</source>
        <translation>立即安裝</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/build.py" line="0"/>
        <source>Keeps this polygon in your session. Export writes all kept polygons to a layer.</source>
        <translation>將此多邊形保留在此工作階段中。Export 會將所有已保留的多邊形寫入圖層。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_maptool.py" line="0"/>
        <source>Keyboard shortcuts</source>
        <translation>鍵盤快速鍵</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>Left-click</source>
        <translation>左鍵點擊</translation>
    </message>
    <message>
        <location filename="../src/core/layer_conventions.py" line="0"/>
        <source>Manual</source>
        <translation>手動</translation>
    </message>
    <message>
        <location filename="../src/core/venv_manager.py" line="0"/>
        <source>OK</source>
        <translation>確定</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_review_build.py" line="0"/>
        <source>One color per object - check neighbors are separated</source>
        <translation>每個物件使用不同顏色－請確認相鄰物件已明確區分</translation>
    </message>
    <message>
        <location filename="../src/ui/plugin/auto_lifecycle.py" line="0"/>
        <source>Out of credits after {done}/{total} tiles. Your detections are kept below.</source>
        <translation>在完成 {done}/{total} 個圖磚後點數已用盡。您的偵測結果會保留於下方。</translation>
    </message>
    <message>
        <location filename="../src/ui/plugin/auto_results.py" line="0"/>
        <source>Outlines only - check boundaries against the imagery</source>
        <translation>僅顯示輪廓－請對照影像確認邊界</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>Pan the map</source>
        <translation>平移地圖</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Part of your zone is outside "{layer}" - only the overlapping area will return objects.</source>
        <translation>您的部分範圍位於「{layer}」之外－僅重疊區域會傳回偵測結果。</translation>
    </message>
    <message>
        <location filename="../src/ui/plugin/auto_run.py" line="0"/>
        <source>Pick an object to detect first (nothing was selected).</source>
        <translation>請先選擇要偵測的物件（目前尚未選取任何物件）。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/state.py" line="0"/>
        <source>Polygon saved ({n} total). Click another element, or export when done.</source>
        <translation>多邊形已儲存（共 {n} 個）。點擊其他地物，或完成後 Export。</translation>
    </message>
    <message>
        <location filename="../src/ui/plugin/auto_results.py" line="0"/>
        <source>Refine seeds</source>
        <translation>細修種子</translation>
    </message>
    <message>
        <location filename="../src/core/checkpoint_manager.py" line="0"/>
        <source>Remove</source>
        <translation>移除</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>Remove area</source>
        <translation>移除區域</translation>
    </message>
    <message>
        <location filename="../src/ui/zone_selection_maptool.py" line="0"/>
        <source>Remove zone</source>
        <translation>移除範圍</translation>
    </message>
    <message>
        <location filename="../src/ui/account_settings_dialog.py" line="0"/>
        <source>Resets {date}</source>
        <translation>於 {date} 重設</translation>
    </message>
    <message>
        <location filename="../src/core/run_restore.py" line="0"/>
        <source>Restored "{prompt}" - adjust and export below.</source>
        <translation>已還原「{prompt}」－請在下方調整並 Export。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>Right-click</source>
        <translation>右鍵點擊</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/build.py" line="0"/>
        <source>Save polygon (S)</source>
        <translation>儲存多邊形（S）</translation>
    </message>
    <message>
        <location filename="../src/ui/plugin/auto_lifecycle.py" line="0"/>
        <source>Session expired. Sign in again to continue.</source>
        <translation>工作階段已逾期，請重新登入以繼續。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>Start (the visible mode's Start button)</source>
        <translation>開始（目前顯示模式的「開始」按鈕）</translation>
    </message>
    <message>
        <location filename="../src/ui/plugin/auto_lifecycle.py" line="0"/>
        <source>The raster was removed. Your polygons were saved to a layer.</source>
        <translation>點陣圖已被移除，您的多邊形已儲存至圖層。</translation>
    </message>
    <message>
        <location filename="../src/ui/plugin/auto_lifecycle.py" line="0"/>
        <source>The selected raster was removed.</source>
        <translation>所選的點陣圖已被移除。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>The selected raster was removed. Keeping what was already found.</source>
        <translation>所選的點陣圖已被移除，已找到的結果會予以保留。</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/about.py" line="0"/>
        <source>Undo the last point</source>
        <translation>復原上一個點</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/state.py" line="0"/>
        <source>Writes a GeoPackage layer with your {n} kept polygons.</source>
        <translation>將您保留的 {n} 個多邊形寫入 GeoPackage 圖層。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>Your zone is outside "{layer}". Pick the right layer or draw inside it.</source>
        <translation>您的範圍位於「{layer}」之外，請選擇正確的圖層或在其範圍內繪製。</translation>
    </message>
    <message>
        <location filename="../src/ui/plugin/auto_zone.py" line="0"/>
        <source>Zone too large. Reduce the area to {max} tiles or fewer.</source>
        <translation>範圍過大，請將範圍縮小至 {max} 個圖磚以內。</translation>
    </message>
    <message>
        <location filename="../src/ui/ai_segmentation_dockwidget.py" line="0"/>
        <source>{n} of {total} free detections left</source>
        <translation>剩餘 {n}／{total} 次免費偵測</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>≈ 1 tile = 1 credit</source>
        <translation>≈ 1 個圖磚＝1 點數</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>all shown</source>
        <translation>全部顯示</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_state.py" line="0"/>
        <source>{visible} of {n} shown</source>
        <translation>已顯示 {visible}／{n}</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_review_build.py" line="0"/>
        <source>all shown</source>
        <translation>全部顯示</translation>
    </message>
    <message>
        <location filename="../src/ui/dock/auto_review_build.py" line="0"/>
        <source>{visible} of {n} shown</source>
        <translation>已顯示 {visible}／{n}</translation>
    </message>
    <message>
        <source>Show guidance tips again</source>
        <translation>再次顯示操作提示</translation>
    </message>
    <message>
        <source>Guidance tips restored</source>
        <translation>操作提示已還原</translation>
    </message>
    <message>
        <source>Run again here</source>
        <translation>在此重新執行</translation>
    </message>
    <message>
        <source>Reload this zone and object, ready to detect.</source>
        <translation>重新載入此範圍與物件，準備進行偵測。</translation>
    </message>
    <message>
        <source>Same object, new zone</source>
        <translation>相同物件，新範圍</translation>
    </message>
    <message>
        <source>Keep this object and draw a new zone on the map.</source>
        <translation>保留此物件，並在地圖上繪製新的範圍。</translation>
    </message>
    <message>
        <source>Upgrade to Pro</source>
        <translation>升級至 Pro</translation>
    </message>
    <message>
        <source>Free account - sign up takes 15 seconds in your browser.</source>
        <translation>免費帳戶－在瀏覽器中註冊僅需 15 秒。</translation>
    </message>
    <message>
        <source>Then segment any imagery: point and click, or fully automatic.</source>
        <translation>接著即可分割任何影像：點擊選取，或完全自動化。</translation>
    </message>
    <message>
        <source>Waiting for your browser sign-in...</source>
        <translation>正在等待您於瀏覽器中登入...</translation>
    </message>
    <message>
        <source>Got it - hide this tip</source>
        <translation>了解了－隱藏此提示</translation>
    </message>
    <message>
        <source>Finish or cancel the current detection before re-running a past one.</source>
        <translation>請先完成或取消目前的偵測，才能重新執行過去的紀錄。</translation>
    </message>
    <message>
        <source>0 shown - lower the Min size filter to reveal them</source>
        <translation>顯示 0 個－降低最小尺寸篩選即可顯示它們</translation>
    </message>
    <message>
        <source>5,000 credits every month. Cancel anytime.</source>
        <translation>每月 5,000 點數。隨時可取消。</translation>
    </message>
    <message>
        <source>A Component Failed to Load</source>
        <translation>元件載入失敗</translation>
    </message>
    <message>
        <source>AI Segmentation</source>
        <translation>AI Segmentation</translation>
    </message>
    <message>
        <source>AI data removed, but some items could not be fully cleared. You can delete the folder manually.</source>
        <translation>AI 資料已移除，但部分項目未能完全清除。您可以手動刪除該資料夾。</translation>
    </message>
    <message>
        <source>Accept the Terms and Privacy Policy first.</source>
        <translation>請先接受服務條款與隱私政策。</translation>
    </message>
    <message>
        <source>An install or detection is still running. Wait for it to finish, then try again.</source>
        <translation>安裝或偵測仍在進行中，請等待完成後再試一次。</translation>
    </message>
    <message>
        <source>Any GeoTIFF, WMS or XYZ basemap.</source>
        <translation>支援任何 GeoTIFF、WMS 或 XYZ 底圖。</translation>
    </message>
    <message>
        <source>Available once the current install or detection finishes.</source>
        <translation>目前的安裝或偵測完成後即可使用。</translation>
    </message>
    <message>
        <source>Blocked by IT Security Policy</source>
        <translation>已被 IT 安全性原則封鎖</translation>
    </message>
    <message>
        <source>Browse the library (view only while detecting).</source>
        <translation>瀏覽範例庫（偵測期間僅供檢視）。</translation>
    </message>
    <message>
        <source>Could not reach the service. Check your connection and try again.</source>
        <translation>無法連線至服務，請檢查您的連線並再試一次。</translation>
    </message>
    <message>
        <source>Could not read pixels from this {ext} file. The file may be corrupt, truncated, or use a compression your GDAL build cannot decode.
Try opening it in QGIS to confirm it displays, or convert it to GeoTIFF (.tif) before using AI Segmentation.</source>
        <translation>無法從此 {ext} 檔案讀取像素。檔案可能已損毀、被截斷，或使用了您的 GDAL 版本無法解碼的壓縮方式。
請嘗試在 QGIS 中開啟以確認是否能正常顯示，或在使用 AI Segmentation 之前將其轉換為 GeoTIFF（.tif）。</translation>
    </message>
    <message>
        <source>Could not remove the AI data. Try again.</source>
        <translation>無法移除 AI 資料，請再試一次。</translation>
    </message>
    <message>
        <source>Couldn't load the demo imagery. Check your internet connection, or add your own layer.</source>
        <translation>無法載入示範影像，請檢查您的網路連線，或新增您自己的圖層。</translation>
    </message>
    <message>
        <source>Detected object</source>
        <translation>已偵測物件</translation>
    </message>
    <message>
        <source>Detection failed. Please try again.</source>
        <translation>偵測失敗，請再試一次。</translation>
    </message>
    <message>
        <source>Downloaded AI data removed. You have been signed out.</source>
        <translation>已移除下載的 AI 資料，您已登出。</translation>
    </message>
    <message>
        <source>Draw on the map</source>
        <translation>在地圖上繪製</translation>
    </message>
    <message>
        <source>Free plan</source>
        <translation>免費方案</translation>
    </message>
    <message>
        <source>Hide parts larger than this ground area. 0 = no limit.</source>
        <translation>隱藏面積大於此地面面積的部分。0＝無限制。</translation>
    </message>
    <message>
        <source>Hide parts smaller than this ground area. Use it to drop tiny noise blobs. 0 = keep all.</source>
        <translation>隱藏面積小於此地面面積的部分，可用於剔除微小的雜訊斑點。0＝全部保留。</translation>
    </message>
    <message>
        <source>Load example imagery</source>
        <translation>載入範例影像</translation>
    </message>
    <message>
        <source>Load your own imagery</source>
        <translation>載入您自己的影像</translation>
    </message>
    <message>
        <source>Lower the Min size filter to show objects first.</source>
        <translation>請先降低最小尺寸篩選以顯示物件。</translation>
    </message>
    <message>
        <source>Manage account</source>
        <translation>管理帳戶</translation>
    </message>
    <message>
        <source>New here? Our 5-minute tutorial walks you through a full detection, step by step.</source>
        <translation>第一次使用嗎？我們的 5 分鐘教學會逐步帶您完成一次完整的偵測。</translation>
    </message>
    <message>
        <source>Not Enough Disk Space</source>
        <translation>磁碟空間不足</translation>
    </message>
    <message>
        <source>Online layer returned blank tiles for this area. The current zoom level may be outside the service's range, or this area has no coverage. Zoom to a level where the layer is visible on the map, then try again.</source>
        <translation>線上圖層在此區域傳回了空白圖磚。目前的縮放層級可能超出該服務的支援範圍，或此區域沒有涵蓋資料。請縮放至圖層能在地圖上顯示的層級後再試一次。</translation>
    </message>
    <message>
        <source>Open the step-by-step tutorial</source>
        <translation>開啟逐步教學</translation>
    </message>
    <message>
        <source>Open the tutorial</source>
        <translation>開啟教學</translation>
    </message>
    <message>
        <source>Opens terra-lab.ai in your browser.</source>
        <translation>在瀏覽器中開啟 terra-lab.ai。</translation>
    </message>
    <message>
        <source>Opens your terra-lab.ai account in the browser.</source>
        <translation>在瀏覽器中開啟您的 terra-lab.ai 帳戶。</translation>
    </message>
    <message>
        <source>Opens your terra-lab.ai dashboard in the browser.</source>
        <translation>在瀏覽器中開啟您的 terra-lab.ai 控制台。</translation>
    </message>
    <message>
        <source>Outline ONE example of the object on the map, then run again. Runs with a drawn example return far fewer empty results.</source>
        <translation>在地圖上勾勒出該物件的一個範例，然後重新執行。附帶繪製範例的執行結果為空的情況會大幅減少。</translation>
    </message>
    <message>
        <source>Planning AI Segmentation run</source>
        <translation>正在規劃 AI Segmentation 執行</translation>
    </message>
    <message>
        <source>Pro plan</source>
        <translation>Pro 方案</translation>
    </message>
    <message>
        <source>Remove downloaded AI data</source>
        <translation>移除已下載的 AI 資料</translation>
    </message>
    <message>
        <source>Remove the downloaded AI data from this computer?</source>
        <translation>要從這台電腦中移除已下載的 AI 資料嗎？</translation>
    </message>
    <message>
        <source>Removing...</source>
        <translation>正在移除...</translation>
    </message>
    <message>
        <source>Right level for {obj} in this zone.</source>
        <translation>此細節層級適合此範圍內的{obj}。</translation>
    </message>
    <message>
        <source>Rotated raster</source>
        <translation>旋轉的點陣圖</translation>
    </message>
    <message>
        <source>Save {save} detections ({hidden} currently hidden by Confidence) to a layer before leaving?</source>
        <translation>離開前是否將 {save} 個偵測結果（其中 {hidden} 個目前因信心度而被隱藏）儲存至圖層？</translation>
    </message>
    <message>
        <source>Save {save} detections to a layer before leaving?</source>
        <translation>離開前是否將 {save} 個偵測結果儲存至圖層？</translation>
    </message>
    <message>
        <source>Segment library (view only)</source>
        <translation>分割範例庫（僅供檢視）</translation>
    </message>
    <message>
        <source>Segmentation failed. Please try again.</source>
        <translation>分割失敗，請再試一次。</translation>
    </message>
    <message>
        <source>Sharper than {obj} usually needs - catches the smallest ones.</source>
        <translation>細節高於{obj}通常所需的程度，可捕捉到最小的物件。</translation>
    </message>
    <message>
        <source>Small {obj} may be missed at this level.</source>
        <translation>在此層級下，較小的{obj}可能會被遺漏。</translation>
    </message>
    <message>
        <source>Something went wrong saving your detections. Please try again.</source>
        <translation>儲存偵測結果時發生錯誤，請再試一次。</translation>
    </message>
    <message>
        <source>Something went wrong starting the detection. Please try again.</source>
        <translation>啟動偵測時發生錯誤，請再試一次。</translation>
    </message>
    <message>
        <source>Support code: {code}</source>
        <translation>支援代碼：{code}</translation>
    </message>
    <message>
        <source>Team or organization?</source>
        <translation>團隊或組織使用？</translation>
    </message>
    <message>
        <source>The AI service is waking up. Holding your spot…</source>
        <translation>AI 服務正在喚醒，正在為您保留名額…</translation>
    </message>
    <message>
        <source>The detection service is busy right now. Please try again in a moment.</source>
        <translation>偵測服務目前忙碌中，請稍後再試。</translation>
    </message>
    <message>
        <source>The service is temporarily unavailable (server error). Your connection is fine - please try again in a few minutes.</source>
        <translation>服務暫時無法使用（伺服器錯誤）。您的連線沒有問題－請幾分鐘後再試一次。</translation>
    </message>
    <message>
        <source>There's a problem with your subscription. Open Settings to update your payment method or review your plan.</source>
        <translation>您的訂閱發生問題，請開啟設定以更新付款方式或查看您的方案。</translation>
    </message>
    <message>
        <source>There's a problem with your subscription. Your last payment may have failed. Open your account to update your payment method or review your plan.</source>
        <translation>您的訂閱發生問題，上次付款可能失敗了。請開啟您的帳戶以更新付款方式或查看您的方案。</translation>
    </message>
    <message>
        <source>This layer has no valid coordinate reference system. Set one in Layer Properties before detecting.</source>
        <translation>此圖層沒有有效的座標參考系統。請在偵測前於圖層屬性中進行設定。</translation>
    </message>
    <message>
        <source>This raster uses a geographic CRS (degrees), which distorts the imagery sent to the AI. For best results, reproject it to a projected CRS (e.g. UTM).</source>
        <translation>此點陣圖使用地理座標系統（度），會導致傳送給 AI 的影像失真。為取得最佳效果，請將其重新投影為投影座標系統（例如 UTM）。</translation>
    </message>
    <message>
        <source>Tip: this raster has no overviews (pyramids). Build them (Raster menu, Miscellaneous, Build Overviews) to make detection much faster.</source>
        <translation>提示：此點陣圖沒有概觀影像（金字塔）。建立概觀影像（點陣圖選單 &gt; 雜項 &gt; 建立概觀影像）可大幅加快偵測速度。</translation>
    </message>
    <message>
        <source>Try "{word}" instead</source>
        <translation>請改用「{word}」</translation>
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
        <translation>有新版本 {version} 可供使用。</translation>
    </message>
    <message>
        <source>Very fine for {obj} - large ones may come back split in parts.</source>
        <translation>對{obj}而言細節過高－較大的物件可能會被拆分成多個部分傳回。</translation>
    </message>
    <message>
        <source>View detections as:</source>
        <translation>偵測結果顯示方式：</translation>
    </message>
    <message>
        <source>We read every message.</source>
        <translation>我們會仔細閱讀每一則訊息。</translation>
    </message>
    <message>
        <source>Write to us:</source>
        <translation>聯絡我們：</translation>
    </message>
    <message>
        <source>Your reference</source>
        <translation>您的參考影像</translation>
    </message>
    <message>
        <source>confident</source>
        <translation>信心度高</translation>
    </message>
    <message>
        <source>polygons</source>
        <translation>多邊形</translation>
    </message>
    <message>
        <source>uncertain</source>
        <translation>信心度低</translation>
    </message>
    <message>
        <source>your object</source>
        <translation>您的物件</translation>
    </message>
    <message>
        <source>{n} found so far</source>
        <translation>目前已發現 {n} 個</translation>
    </message>
    <!-- v2.1.7 sync: append-to-layer export, singular forms, small-example guidance (2026-07-13) -->
    <message>
        <source>5,000 detections every month</source>
        <translation>每月 5,000 次偵測（約 1,700 平方公里）</translation>
    </message>
    <message>
        <source>This zone is {area} km². Free zones stop at {max} km².</source>
        <translation>這個區域為 {area} km²。免費區域最大 {max} km²。</translation>
    </message>
    <message>
        <source>Pro has no size limit. Any area you draw, 5,000 tiles a month, maximum detail.</source>
        <translation>Pro 沒有大小限制。你畫多大都行，每月 5,000 個圖磚，最高細節。</translation>
    </message>
    <message>
        <source>&lt;a href=&quot;{url}&quot;&gt;Upgrade to Pro&lt;/a&gt;, or make this zone smaller.</source>
        <translation>&lt;a href=&quot;{url}&quot;&gt;升級到 Pro&lt;/a&gt;，或把區域畫小一點。</translation>
    </message>
    <message>
        <source>Sending to the AI...</source>
        <translation>正在傳送至 AI...</translation>
    </message>
    <message>
        <source>Spot reserved · starting in a few seconds...</source>
        <translation>已保留名額·數秒後開始...</translation>
    </message>
    <message>
        <source>Spot reserved · starting soon...</source>
        <translation>已保留名額·即將開始...</translation>
    </message>
    <message>
        <source>Stopping - keeping the tiles already found...</source>
        <translation>正在停止－已發現的圖磚將予以保留...</translation>
    </message>
    <message>
        <source>Stopping...</source>
        <translation>正在停止...</translation>
    </message>
    <message>
        <source>The AI is starting up, almost there... {n}s</source>
        <translation>AI 正在啟動，即將就緒... {n} 秒</translation>
    </message>
    <message>
        <source>Waking up the AI... {n}s</source>
        <translation>正在喚醒 AI... {n} 秒</translation>
    </message>
    <message>
        <source>You're next · starting now...</source>
        <translation>輪到您了·即將開始...</translation>
    </message>
    <message>
        <source>Cancelled</source>
        <translation>已取消</translation>
    </message>
    <message>
        <source>Free trial</source>
        <translation>免費試用</translation>
    </message>
    <message>
        <source>Select a raster layer to segment:</source>
        <translation>選擇要分割的點陣圖圖層：</translation>
    </message>
    <message>
        <source>Your {n} free detections are used up</source>
        <translation>您的 {n} 次免費偵測已用完</translation>
    </message>
    <message>
        <source>1 object found</source>
        <translation>找到 1 個物件</translation>
    </message>
    <message>
        <source>Download AI model</source>
        <translation>下載 AI 模型</translation>
    </message>
    <message>
        <source>Export 1 polygon</source>
        <translation>Export 1 個多邊形</translation>
    </message>
    <message>
        <source>Resolving object name</source>
        <translation>正在解析物件名稱</translation>
    </message>
    <message>
        <source>Your free detections are used up</source>
        <translation>您的免費偵測已用完</translation>
    </message>
    <message>
        <source>&quot;{obj}&quot; is not an object the AI knows well. Drawing one example on the map shows it what to find.</source>
        <translation>「{obj}」不是 AI 熟悉的物件。在地圖上繪製一個範例，讓它知道要尋找什麼。</translation>
    </message>
    <message>
        <source>&quot;{obj}&quot; is often missed from text alone. Draw one example on the map to find far more.</source>
        <translation>單靠文字時，「{obj}」常常會被漏掉。在地圖上繪製一個範例，可以找到更多。</translation>
    </message>
    <message>
        <source>1 correction this round</source>
        <translation>本輪 1 次修正</translation>
    </message>
    <message>
        <source>1 object</source>
        <translation>1 個物件</translation>
    </message>
    <message>
        <source>1 polygon added so far.</source>
        <translation>目前已新增 1 個多邊形。</translation>
    </message>
    <message>
        <source>1 result</source>
        <translation>1 筆結果</translation>
    </message>
    <message>
        <source>A newer version of AI Segmentation is available with the latest fixes.</source>
        <translation>有新版本的 AI Segmentation 可供使用，內含最新修正。</translation>
    </message>
    <message>
        <source>A shape was removed. Click Save to confirm.</source>
        <translation>已移除一個形狀，請點擊「儲存」以確認。</translation>
    </message>
    <message>
        <source>AI</source>
        <translation>AI</translation>
    </message>
    <message>
        <source>AI Environment Damaged</source>
        <translation>AI 環境已損毀</translation>
    </message>
    <message>
        <source>Add a missing polygon</source>
        <translation>新增遺漏的多邊形</translation>
    </message>
    <message>
        <source>Add an object the AI missed. In AI, point at it and the on-device model outlines it, free; in Manual, draw its corners.</source>
        <translation>新增 AI 遺漏的物件：AI 方式下，點一下即可由本機模型免費勾勒輪廓；手動方式下，自行繪製其角點。</translation>
    </message>
    <message>
        <source>Add another example - more references detect more</source>
        <translation>新增另一個範例－更多參考影像可偵測更多</translation>
    </message>
    <message>
        <source>Add one more example for the best results.</source>
        <translation>再新增一個範例以獲得最佳結果。</translation>
    </message>
    <message>
        <source>Adding needs a one-time setup</source>
        <translation>新增需要進行一次性設定</translation>
    </message>
    <message>
        <source>Almost done - building the shapes...</source>
        <translation>即將完成－正在建立形狀...</translation>
    </message>
    <message>
        <source>Another QGIS window is installing the AI components. Wait for it to finish, then try again.</source>
        <translation>另一個 QGIS 視窗正在安裝 AI 元件，請等待其完成後再試一次。</translation>
    </message>
    <message>
        <source>Another QGIS window is installing the AI engine. Wait for it to finish, then try again.</source>
        <translation>另一個 QGIS 視窗正在安裝 AI 引擎，請等待其完成後再試一次。</translation>
    </message>
    <message>
        <source>Automatic detection failed</source>
        <translation>自動偵測失敗</translation>
    </message>
    <message>
        <source>Automatic: detect</source>
        <translation>自動：偵測</translation>
    </message>
    <message>
        <source>Automatic: draw the zone</source>
        <translation>自動：繪製範圍</translation>
    </message>
    <message>
        <source>Automatic: merge with neighbours</source>
        <translation>自動：與鄰近物件合併</translation>
    </message>
    <message>
        <source>Automatic: review and Correct</source>
        <translation>自動：檢視並修正</translation>
    </message>
    <message>
        <source>Best quality. Two references locked in.</source>
        <translation>最佳品質，已鎖定兩個參考影像。</translation>
    </message>
    <message>
        <source>Blocked by Antivirus or Security Software</source>
        <translation>已被防毒軟體或安全性軟體封鎖</translation>
    </message>
    <message>
        <source>Calculating...</source>
        <translation>正在計算...</translation>
    </message>
    <message>
        <source>Cancel the example box, the detection, or exit Automatic</source>
        <translation>取消範例框或偵測，或退出自動模式</translation>
    </message>
    <message>
        <source>Cancel the merge</source>
        <translation>取消合併</translation>
    </message>
    <message>
        <source>Change recorded.</source>
        <translation>已記錄變更。</translation>
    </message>
    <message>
        <source>Checking the object name</source>
        <translation>正在檢查物件名稱</translation>
    </message>
    <message>
        <source>Checking the object name...</source>
        <translation>正在檢查物件名稱...</translation>
    </message>
    <message>
        <source>Choose how to fix the polygon: AI points or QGIS vertices</source>
        <translation>選擇修正多邊形的方式：AI 點選或 QGIS 頂點</translation>
    </message>
    <message>
        <source>Clean up the outlines</source>
        <translation>清理輪廓</translation>
    </message>
    <message>
        <source>Clear all</source>
        <translation>全部清除</translation>
    </message>
    <message>
        <source>Clear the points, then exit Automatic</source>
        <translation>清除已放置的點，然後退出自動模式</translation>
    </message>
    <message>
        <source>Clear the selection, or stop the segmentation</source>
        <translation>清除選取範圍，或停止分割</translation>
    </message>
    <message>
        <source>Click a polygon, then click the spot the AI missed.</source>
        <translation>點擊一個多邊形，再點擊 AI 遺漏的位置。</translation>
    </message>
    <message>
        <source>Click a polygon, then drag any corner.</source>
        <translation>點擊一個多邊形，再拖曳任一角點。</translation>
    </message>
    <message>
        <source>Click an object on the map and the AI outlines it.</source>
        <translation>在地圖上點擊一個物件，AI 就會勾勒出其輪廓。</translation>
    </message>
    <message>
        <source>Click each corner of the object, then Finish.</source>
        <translation>依序點擊物件的每個角點，再點擊「完成」。</translation>
    </message>
    <message>
        <source>Click each corner on the map, then Finish the line.</source>
        <translation>依序在地圖上點擊每個角點，然後完成這條線。</translation>
    </message>
    <message>
        <source>Click each piece of the object you want to join.</source>
        <translation>依序點擊要合併的每個物件部分。</translation>
    </message>
    <message>
        <source>Click to open this step</source>
        <translation>點擊以開啟此步驟</translation>
    </message>
    <message>
        <source>Close the fix, clear the selection, or exit the review</source>
        <translation>關閉修正、清除選取範圍，或退出檢視</translation>
    </message>
    <message>
        <source>Close the gaps inside this polygon, without filling the courtyards the rest of the layer is meant to keep.</source>
        <translation>封閉此多邊形內部的細縫，不影響圖層其他部分刻意保留的中庭。</translation>
    </message>
    <message>
        <source>Close the line you are drawing. A right-click on the map does the same.</source>
        <translation>封閉您正在繪製的線，在地圖上按右鍵也有相同效果。</translation>
    </message>
    <message>
        <source>Closes the hairline gaps between neighbouring shapes, for land cover maps.</source>
        <translation>封閉相鄰形狀間的細縫，適用於土地覆蓋圖。</translation>
    </message>
    <message>
        <source>Confirm the merge</source>
        <translation>確認合併</translation>
    </message>
    <message>
        <source>Correct</source>
        <translation>修正</translation>
    </message>
    <message>
        <source>Could not apply the new settings. Try a different value.</source>
        <translation>無法套用新設定，請嘗試不同的數值。</translation>
    </message>
    <message>
        <source>Could not check the AI components. See the log for details.</source>
        <translation>無法檢查 AI 元件，詳情請見日誌。</translation>
    </message>
    <message>
        <source>Credits come back on {date}</source>
        <translation>點數將於 {date} 恢復</translation>
    </message>
    <message>
        <source>Crop fetch was cancelled.</source>
        <translation>影像裁切已取消。</translation>
    </message>
    <message>
        <source>Cut thin spurs off this polygon (0 = off). Raise it on a single ragged outline instead of eroding the whole layer.</source>
        <translation>修剪此多邊形上的細小尖刺（0＝關閉）。可只針對單一鋸齒狀輪廓調高數值，而不必侵蝕整個圖層。</translation>
    </message>
    <message>
        <source>Delete this corner</source>
        <translation>刪除此角點</translation>
    </message>
    <message>
        <source>Delete this polygon</source>
        <translation>刪除此多邊形</translation>
    </message>
    <message>
        <source>Delete this polygon (the Delete key works too, and a right-click on the map deletes the shape under the cursor). Undo brings it back.</source>
        <translation>刪除此多邊形（也可以按 Delete 鍵，或在地圖上按右鍵刪除游標所在的形狀）。復原可將其還原。</translation>
    </message>
    <message>
        <source>Deleting the downloaded data...</source>
        <translation>正在刪除已下載的資料...</translation>
    </message>
    <message>
        <source>Dense area {current}/{total}</source>
        <translation>密集區域 {current}/{total}</translation>
    </message>
    <message>
        <source>Dense forest? &quot;Forest&quot; takes it as one block; &quot;Tree&quot; picks individual trees.</source>
        <translation>森林很密集嗎？「Forest」會將其視為一整塊；「Tree」則會挑出個別樹木。</translation>
    </message>
    <message>
        <source>Detection stopped early after {done} tile(s). The objects already found are kept below.</source>
        <translation>偵測在完成 {done} 個圖磚後提前停止，已找到的物件會保留在下方。</translation>
    </message>
    <message>
        <source>Discard reviewed results and run again? Confirm</source>
        <translation>捨棄已檢視的結果並重新執行？確認</translation>
    </message>
    <message>
        <source>Distinct</source>
        <translation>獨立</translation>
    </message>
    <message>
        <source>Drag a corner to move it. Click an edge to add one, right-click removes.</source>
        <translation>拖曳角點以移動，點擊邊緣可新增角點，右鍵點擊可移除。</translation>
    </message>
    <message>
        <source>Drag a corner to move it. Double-click an edge to add one.</source>
        <translation>拖曳角點以移動，雙擊邊緣可新增角點。</translation>
    </message>
    <message>
        <source>Drag, add or delete the object&apos;s corners by hand.</source>
        <translation>手動拖曳、新增或刪除物件的角點。</translation>
    </message>
    <message>
        <source>Draw a line across the object to cut it into two.</source>
        <translation>在物件上繪製一條線，將其分割成兩個。</translation>
    </message>
    <message>
        <source>Draw a line right across the shape, then Finish.</source>
        <translation>在形狀上繪製一條貫穿的線，再點擊「完成」。</translation>
    </message>
    <message>
        <source>Draw an example, or type what to find.</source>
        <translation>繪製範例，或輸入要尋找的內容。</translation>
    </message>
    <message>
        <source>Draw its corners</source>
        <translation>繪製其角點</translation>
    </message>
    <message>
        <source>Draw one &apos;{object}&apos; - the AI finds the rest</source>
        <translation>繪製一個「{object}」，AI 會找出其餘的</translation>
    </message>
    <message>
        <source>Draw one example - the AI finds the rest</source>
        <translation>繪製一個範例，AI 會找出其餘的</translation>
    </message>
    <message>
        <source>Draw the new edge: start outside the shape, cross it, end outside, then Finish.</source>
        <translation>繪製新邊緣：從形狀外部開始，穿過形狀，在外部結束，再點擊「完成」。</translation>
    </message>
    <message>
        <source>Drawn examples</source>
        <translation>已繪製的範例</translation>
    </message>
    <message>
        <source>Drop points closer than this distance to a straight edge (0 = off). A distance, not a count: pushed high it can flatten curved walls. Points is usually the better dial for thinning an outline.</source>
        <translation>移除距離直線邊緣小於此距離的節點（0＝關閉）。這是距離，而非數量：調得太高可能會壓平彎曲的牆面。若要精簡輪廓，通常「節點」是較好的控制項。</translation>
    </message>
    <message>
        <source>Drop points closer than this distance to a straight edge (0 = off). A distance, not a count: pushed high it can flatten curved walls. Points is usually the better dial; this stays for comparison.</source>
        <translation>移除距離直線邊緣小於此距離的節點（0＝關閉）。這是距離，而非數量：調得太高可能會壓平彎曲的牆面。通常「節點」是較好的控制項，此項目僅供比較保留。</translation>
    </message>
    <message>
        <source>Drop this polygon&apos;s points closer than this distance to a straight edge (0 = off). A distance, not a count; Points is usually the better dial.</source>
        <translation>移除此多邊形上距離直線邊緣小於此距離的節點（0＝關閉）。這是距離，而非數量；通常「節點」是較好的控制項。</translation>
    </message>
    <message>
        <source>Edit an existing polygon</source>
        <translation>編輯現有的多邊形</translation>
    </message>
    <message>
        <source>Export the polygons to a layer</source>
        <translation>Export 多邊形至圖層</translation>
    </message>
    <message>
        <source>Fewer points</source>
        <translation>較少節點</translation>
    </message>
    <message>
        <source>Fill holes</source>
        <translation>填補孔洞</translation>
    </message>
    <message>
        <source>Fill only holes smaller than this ground area. Bigger holes (a road median, a courtyard) stay open. No limit = fill every hole.</source>
        <translation>僅填補小於此地面面積的孔洞，較大的孔洞（如道路中央分隔島、中庭）會保持開放。無限制＝填補所有孔洞。</translation>
    </message>
    <message>
        <source>Finish the line</source>
        <translation>完成這條線</translation>
    </message>
    <message>
        <source>Fix method</source>
        <translation>修正方式</translation>
    </message>
    <message>
        <source>Give neighbouring shapes one exact border instead of a hairline gap or overlap. For land cover, where the map is one surface.</source>
        <translation>讓相鄰形狀共用同一條精確邊界，而非留下細縫或重疊。適用於地圖為單一表面的土地覆蓋資料。</translation>
    </message>
    <message>
        <source>Give this one polygon its own shape settings, without moving the dials that drive the whole layer.</source>
        <translation>讓這個多邊形套用專屬的形狀設定，而不影響驅動整個圖層的控制項。</translation>
    </message>
    <message>
        <source>Go back to picking polygons. Everything you kept stays, and so does the outline on screen.</source>
        <translation>返回多邊形選取畫面。您保留的內容都會保留，畫面上的輪廓也會保留。</translation>
    </message>
    <message>
        <source>Go back to your zone, references and settings, then detect the whole zone again. Nothing is saved.</source>
        <translation>回到您的範圍、參考影像和設定，然後重新偵測整個範圍。系統不會儲存任何內容。</translation>
    </message>
    <message>
        <source>Grow / shrink</source>
        <translation>擴張／收縮</translation>
    </message>
    <message>
        <source>How detections are coloured on the map (visual only): Normal fill, Outline, Confidence heatmap, or a distinct colour per object to tell them apart.</source>
        <translation>地圖上偵測結果的著色方式（僅影響顯示）：一般填色、輪廓、信心度熱力圖，或依物件套用獨立顏色以利區分。</translation>
    </message>
    <message>
        <source>How many of this polygon&apos;s points to keep. The count in the title row follows it. It runs before Right angles, so lowering it gives the squaring straight walls instead of a staircase.</source>
        <translation>這個多邊形要保留多少節點。標題列中的數量會隨之更新。此設定會在「直角化」之前執行，因此調低它可讓直角化後的牆面更平直，而非階梯狀。</translation>
    </message>
    <message>
        <source>How sure the AI is about each object. Lower shows more, higher keeps only the sure ones.</source>
        <translation>AI 對每個物件的把握程度。數值越低顯示越多，數值越高則只保留把握最大的物件。</translation>
    </message>
    <message>
        <source>Identify new shape</source>
        <translation>識別新形狀</translation>
    </message>
    <message>
        <source>Installation Already Running</source>
        <translation>安裝已在執行中</translation>
    </message>
    <message>
        <source>Installation Path Problem</source>
        <translation>安裝路徑問題</translation>
    </message>
    <message>
        <source>Installation running in another window</source>
        <translation>安裝正在另一個視窗中執行</translation>
    </message>
    <message>
        <source>Keep</source>
        <translation>保留</translation>
    </message>
    <message>
        <source>Keep these edits and go back to picking polygons.</source>
        <translation>保留這些編輯，並返回多邊形選取畫面。</translation>
    </message>
    <message>
        <source>Keep this detection in Favorites</source>
        <translation>收藏此偵測結果</translation>
    </message>
    <message>
        <source>Keep this object in Favorites</source>
        <translation>收藏此物件</translation>
    </message>
    <message>
        <source>Keep this one</source>
        <translation>保留這個</translation>
    </message>
    <message>
        <source>Keep this one, or click again to correct the outline.</source>
        <translation>保留這個，或再次點擊以修正輪廓。</translation>
    </message>
    <message>
        <source>Keep this one, or keep placing corners.</source>
        <translation>保留這個，或繼續放置角點。</translation>
    </message>
    <message>
        <source>Keep this outline and point at the next object. Shortcut: S</source>
        <translation>保留此輪廓，並點選下一個物件。快速鍵：S</translation>
    </message>
    <message>
        <source>Left-click adds a keep point, right-click a trim point. The outline follows.</source>
        <translation>左鍵點擊新增保留點，右鍵點擊新增修剪點，輪廓會隨之調整。</translation>
    </message>
    <message>
        <source>Linux System Too Old</source>
        <translation>Linux 系統版本過舊</translation>
    </message>
    <message>
        <source>Loading stored detections ({done} of {total})</source>
        <translation>正在載入已儲存的偵測結果（{done}/{total}）</translation>
    </message>
    <message>
        <source>Maximum</source>
        <translation>最大值</translation>
    </message>
    <message>
        <source>Measuring AI data size...</source>
        <translation>正在測量 AI 資料大小...</translation>
    </message>
    <message>
        <source>Measuring the downloaded data...</source>
        <translation>正在測量已下載的資料...</translation>
    </message>
    <message>
        <source>Merge with neighbours</source>
        <translation>與鄰近物件合併</translation>
    </message>
    <message>
        <source>Merge {n} shapes · Free</source>
        <translation>合併 {n} 個形狀·免費</translation>
    </message>
    <message>
        <source>Minimum</source>
        <translation>最小值</translation>
    </message>
    <message>
        <source>Missing System Component</source>
        <translation>缺少系統元件</translation>
    </message>
    <message>
        <source>Move points</source>
        <translation>移動角點</translation>
    </message>
    <message>
        <source>Navigation (while a tool is armed)</source>
        <translation>導覽（工具啟用時）</translation>
    </message>
    <message>
        <source>New polygon</source>
        <translation>新多邊形</translation>
    </message>
    <message>
        <source>New shape added. Click Save to keep it.</source>
        <translation>已新增一個形狀，請點擊「儲存」以保留。</translation>
    </message>
    <message>
        <source>No connection to the sign-in service. Check your internet connection, then click Connect to try again.</source>
        <translation>無法連線至登入服務，請檢查您的網路連線，然後點擊「連線」再試一次。</translation>
    </message>
    <message>
        <source>No detection under that click.</source>
        <translation>該點擊位置沒有偵測結果。</translation>
    </message>
    <message>
        <source>No matches in this zone.</source>
        <translation>此範圍內沒有相符結果。</translation>
    </message>
    <message>
        <source>No object matches that search.</source>
        <translation>沒有物件符合該搜尋條件。</translation>
    </message>
    <message>
        <source>Nothing changed. The line has to cross the outline twice, starting and ending outside the shape.</source>
        <translation>沒有變更。這條線必須兩次穿過輪廓，起點與終點都要在形狀外部。</translation>
    </message>
    <message>
        <source>Nothing was added. A polygon needs at least three corners.</source>
        <translation>未新增任何內容，多邊形至少需要三個角點。</translation>
    </message>
    <message>
        <source>Nothing was split. The line has to cross the shape completely, starting and ending outside it.</source>
        <translation>未進行分割，這條線必須完全穿過形狀，起點與終點都要在形狀外部。</translation>
    </message>
    <message>
        <source>Now click the other pieces of this object.</source>
        <translation>現在請點擊此物件的其他部分。</translation>
    </message>
    <message>
        <source>One object came back split into several polygons. Click the others on the map, then confirm to merge them into one.</source>
        <translation>有一個物件被拆分成多個多邊形。請在地圖上點擊其餘部分，再確認以合併為一個物件。</translation>
    </message>
    <message>
        <source>One object per run - Detect will run &quot;{first}&quot; first.</source>
        <translation>每次執行僅能偵測一種物件－系統會先執行「{first}」。</translation>
    </message>
    <message>
        <source>One object per run - detecting &quot;{first}&quot; now. Run the other objects as separate detections.</source>
        <translation>每次執行僅能偵測一種物件－目前正在偵測「{first}」。其他物件請另外執行偵測。</translation>
    </message>
    <message>
        <source>Open the selected saved polygon for AI editing</source>
        <translation>開啟所選的已儲存多邊形進行 AI 編輯</translation>
    </message>
    <message>
        <source>Other</source>
        <translation>其他</translation>
    </message>
    <message>
        <source>Package Versions Conflict</source>
        <translation>套件版本衝突</translation>
    </message>
    <message>
        <source>Pick a tool above, then edit the highlighted object.</source>
        <translation>請於上方選擇工具，再編輯醒目顯示的物件。</translation>
    </message>
    <message>
        <source>Pick at least two shapes to merge them.</source>
        <translation>請至少選擇兩個形狀才能合併。</translation>
    </message>
    <message>
        <source>Pick or un-pick an object</source>
        <translation>選取或取消選取物件</translation>
    </message>
    <message>
        <source>Point at it on the map</source>
        <translation>在地圖上點選它</translation>
    </message>
    <message>
        <source>Points</source>
        <translation>節點</translation>
    </message>
    <message>
        <source>Points the map back at this run with the same object and the same number of tiles, ready to detect. Nothing is spent until you do.</source>
        <translation>將地圖重新指向此次執行，使用相同物件與相同圖磚數量，準備好即可偵測。在您執行前不會耗費任何點數。</translation>
    </message>
    <message>
        <source>Points: {n}</source>
        <translation>節點：{n}</translation>
    </message>
    <message>
        <source>Positive = grow outward, negative = shrink inward</source>
        <translation>正值＝向外擴張，負值＝向內收縮</translation>
    </message>
    <message>
        <source>Proxy Authentication Required</source>
        <translation>需要代理伺服器驗證</translation>
    </message>
    <message>
        <source>Push this polygon&apos;s edge out (positive) or in (negative), for the one footprint the model cut short or overran.</source>
        <translation>將此多邊形的邊緣向外（正值）或向內（負值）推移，適用於模型判斷範圍過小或過大的單一物件。</translation>
    </message>
    <message>
        <source>QGIS could not activate the temporary review layer. Close any other editing session, then try Edit manually again.</source>
        <translation>QGIS 無法啟用暫存檢視圖層，請關閉其他編輯工作階段，再重新嘗試手動編輯。</translation>
    </message>
    <message>
        <source>QGIS could not save these edits. Fix the geometry and click Done again.</source>
        <translation>QGIS 無法儲存這些編輯，請修正幾何圖形，再次點擊「確定」。</translation>
    </message>
    <message>
        <source>Re-run the whole zone</source>
        <translation>重新執行整個範圍</translation>
    </message>
    <message>
        <source>Reading the imagery around this polygon...</source>
        <translation>正在讀取此多邊形周圍的影像...</translation>
    </message>
    <message>
        <source>Reading the imagery around your click...</source>
        <translation>正在讀取您點擊位置周圍的影像...</translation>
    </message>
    <message>
        <source>Reading this run...</source>
        <translation>正在讀取此次執行...</translation>
    </message>
    <message>
        <source>Rebuilding shapes ({done} of {total})</source>
        <translation>正在重建形狀（{done}/{total}）</translation>
    </message>
    <message>
        <source>Redraw edge</source>
        <translation>重新繪製邊緣</translation>
    </message>
    <message>
        <source>Remove the corner you picked. The Delete key does the same.</source>
        <translation>移除您所選取的角點，按 Delete 鍵也有相同效果。</translation>
    </message>
    <message>
        <source>Remove the selected detection</source>
        <translation>移除所選的偵測結果</translation>
    </message>
    <message>
        <source>Removing the downloaded AI data. This window closes when it is done.</source>
        <translation>正在移除已下載的 AI 資料，完成後此視窗會自動關閉。</translation>
    </message>
    <message>
        <source>Removing the downloaded AI data...</source>
        <translation>正在移除已下載的 AI 資料...</translation>
    </message>
    <message>
        <source>Replace one side by drawing a new line across the outline.</source>
        <translation>繪製一條貫穿輪廓的新線條，以取代其中一側。</translation>
    </message>
    <message>
        <source>Report this problem</source>
        <translation>回報此問題</translation>
    </message>
    <message>
        <source>Reset to shared</source>
        <translation>重設為共用設定</translation>
    </message>
    <message>
        <source>Right angles</source>
        <translation>直角化</translation>
    </message>
    <message>
        <source>Round corners</source>
        <translation>圓角</translation>
    </message>
    <message>
        <source>Round corners for natural shapes like trees and bushes. Lower Points for smoother results.</source>
        <translation>為樹木、灌木等自然形狀套用圓角，降低「節點」可讓結果更平滑。</translation>
    </message>
    <message>
        <source>Round this polygon&apos;s corners, for a tree or a pond among squared neighbours.</source>
        <translation>為此多邊形套用圓角，適合在方正的鄰近物件中呈現樹木或池塘等形狀。</translation>
    </message>
    <message>
        <source>Run the detection</source>
        <translation>執行偵測</translation>
    </message>
    <message>
        <source>Run this zone again</source>
        <translation>重新執行此範圍</translation>
    </message>
    <message>
        <source>Save</source>
        <translation>儲存</translation>
    </message>
    <message>
        <source>Save the fix and go back to the review</source>
        <translation>儲存修正並返回檢視</translation>
    </message>
    <message>
        <source>Shape updated. Keep editing, or click Save.</source>
        <translation>形狀已更新，可繼續編輯，或點擊「儲存」。</translation>
    </message>
    <message>
        <source>Shapes</source>
        <translation>形狀</translation>
    </message>
    <message>
        <source>Shared borders:</source>
        <translation>共用邊界：</translation>
    </message>
    <message>
        <source>Shave thin spikes and ragged bits off each shape&apos;s outline. It leaves the main body alone; higher values trim more. 0 = off.</source>
        <translation>修剪每個形狀輪廓上的細小尖刺與鋸齒狀部分，主體不受影響；數值越高修剪越多。0＝關閉。</translation>
    </message>
    <message>
        <source>Show what it looks like</source>
        <translation>顯示外觀</translation>
    </message>
    <message>
        <source>Simplify</source>
        <translation>簡化</translation>
    </message>
    <message>
        <source>Snap walls to right angles, 45 degree walls included. Made for buildings, pools and solar panels. A shape it would distort is left as it is.</source>
        <translation>將牆面校正為直角，包含 45 度角的牆面。適用於建築物、泳池、太陽能板。若校正會使形狀失真，則會保持原樣。</translation>
    </message>
    <message>
        <source>Something went wrong preparing the results. Please run Detect again.</source>
        <translation>準備結果時發生問題，請重新執行偵測。</translation>
    </message>
    <message>
        <source>Something went wrong preparing the results. Your detections were saved to the layer {name}.</source>
        <translation>準備結果時發生問題，您的偵測結果已儲存至圖層 {name}。</translation>
    </message>
    <message>
        <source>Split</source>
        <translation>分割</translation>
    </message>
    <message>
        <source>Square this polygon&apos;s edges, or leave them as traced while the rest of the layer stays squared.</source>
        <translation>將此多邊形的邊緣校正為直角，或維持原始描繪的形狀，即使圖層其餘部分仍為直角化。</translation>
    </message>
    <message>
        <source>Star a detection or an object to keep it here.</source>
        <translation>將偵測結果或物件加入收藏，即可保留在這裡。</translation>
    </message>
    <message>
        <source>Still waiting for the sign-in page. If no browser opened, or the page shows an error, click Cancel and try again.</source>
        <translation>仍在等待登入頁面。若瀏覽器未開啟，或頁面顯示錯誤，請點擊「取消」再試一次。</translation>
    </message>
    <message>
        <source>Stop adding</source>
        <translation>停止新增</translation>
    </message>
    <message>
        <source>Stopping the local AI...</source>
        <translation>正在停止本機 AI...</translation>
    </message>
    <message>
        <source>That area does not touch the object you are editing, so nothing was added. Reshaping works on one object at a time.</source>
        <translation>該區域未接觸到您正在編輯的物件，因此未新增任何內容。重塑一次僅能處理一個物件。</translation>
    </message>
    <message>
        <source>That ground belongs to another object, so nothing was added. Edit that object instead, or join the two with Merge with neighbours.</source>
        <translation>該區域屬於另一個物件，因此未新增任何內容。請改為編輯該物件，或使用「與鄰近物件合併」將兩者合併。</translation>
    </message>
    <message>
        <source>That is another object. The one you were editing is saved, and this one is now selected.</source>
        <translation>這是另一個物件。您原本編輯的物件已儲存，現在已選取這一個。</translation>
    </message>
    <message>
        <source>The AI finds every object that looks like your examples - you can draw up to 3.</source>
        <translation>AI 會找出所有與您範例相似的物件－最多可繪製 3 個範例。</translation>
    </message>
    <message>
        <source>The AI outlines it, free, on your computer.</source>
        <translation>AI 會在您的電腦上免費為其勾勒輪廓。</translation>
    </message>
    <message>
        <source>The connection to the server was interrupted. Please try again.</source>
        <translation>與伺服器的連線已中斷，請再試一次。</translation>
    </message>
    <message>
        <source>The detection service had a problem and the run stopped. Please try again.</source>
        <translation>偵測服務發生問題，執行已停止，請再試一次。</translation>
    </message>
    <message>
        <source>The detection stopped responding. Keeping the {n} tiles already found.</source>
        <translation>偵測已停止回應，已找到的 {n} 個圖磚將保留。</translation>
    </message>
    <message>
        <source>The imagery reader could not be loaded, and repairing the installation did not fix it. Please report this so we can look into it.

{details}</source>
        <translation>無法載入影像讀取工具，且修復安裝也未能解決問題。請回報此問題，以便我們進行調查。

{details}</translation>
    </message>
    <message>
        <source>The installer could not start a helper process (a damaged Python launcher). Click Reinstall Dependencies to rebuild the environment from scratch.</source>
        <translation>安裝程式無法啟動輔助處理程序（Python 啟動器已損毀）。請點擊「重新安裝相依套件」以重新建立環境。</translation>
    </message>
    <message>
        <source>The removal could not start. You are signed out, but the downloaded AI data is still on this computer. Try again.</source>
        <translation>無法開始移除。您已登出，但已下載的 AI 資料仍在此電腦上，請再試一次。</translation>
    </message>
    <message>
        <source>The removal did not finish. Close this window, then check the AI data folder before trying again.</source>
        <translation>移除未完成，請關閉此視窗，並在再試一次之前先檢查 AI 資料夾。</translation>
    </message>
    <message>
        <source>The removal is already running.</source>
        <translation>移除作業已在執行中。</translation>
    </message>
    <message>
        <source>The reply did not come from the service. If this network shows a sign-in page, open it in your browser first, then try again.</source>
        <translation>回應並非來自該服務。若此網路會顯示登入頁面，請先在瀏覽器中開啟該頁面，再試一次。</translation>
    </message>
    <message>
        <source>The server returned an unexpected response. Please try again.</source>
        <translation>伺服器傳回非預期的回應，請再試一次。</translation>
    </message>
    <message>
        <source>Thin this polygon&apos;s points before you edit them by hand. 100% keeps the outline as it is.</source>
        <translation>在手動編輯前先精簡此多邊形的節點。100% 會維持輪廓原狀。</translation>
    </message>
    <message>
        <source>This polygon</source>
        <translation>此多邊形</translation>
    </message>
    <message>
        <source>This raster has no coordinate reference system, so polygons will use pixel coordinates. Set a CRS in Layer Properties for georeferenced output.</source>
        <translation>此點陣圖沒有座標參考系統，因此多邊形將使用像素座標。請在圖層屬性中設定 CRS，以取得具地理參照的輸出結果。</translation>
    </message>
    <message>
        <source>This raster is rotated. Run Warp (Reproject) on it to straighten it before segmenting.</source>
        <translation>此點陣圖已旋轉。請先執行「Warp（重新投影）」將其校正，再進行分割。</translation>
    </message>
    <message>
        <source>This run did not keep where it looked, so it cannot be pointed at the same place. Draw the zone again.</source>
        <translation>此次執行未保留偵測範圍的位置資訊，因此無法重新指向相同地點，請重新繪製範圍。</translation>
    </message>
    <message>
        <source>Those shapes could not be joined. Nothing was changed.</source>
        <translation>這些形狀無法合併，未進行任何變更。</translation>
    </message>
    <message>
        <source>Trim spikes</source>
        <translation>修剪尖刺</translation>
    </message>
    <message>
        <source>Two references give the strongest detection. Draw a second to reach best quality.</source>
        <translation>兩個參考影像可提供最強的偵測效果，請再繪製第二個以達到最佳品質。</translation>
    </message>
    <message>
        <source>Unavailable while Right angles is on. Turn it off to adjust this setting.</source>
        <translation>開啟「直角化」時無法使用，請關閉後再調整此設定。</translation>
    </message>
    <message>
        <source>Undo</source>
        <translation>復原</translation>
    </message>
    <message>
        <source>Undo last</source>
        <translation>復原上一步</translation>
    </message>
    <message>
        <source>Undo the last correction</source>
        <translation>復原上一次修正</translation>
    </message>
    <message>
        <source>Undo the last thing you did here: the point you just placed, or the last edit.</source>
        <translation>復原您剛才在此處的操作：剛放置的點，或上一次編輯。</translation>
    </message>
    <message>
        <source>Unsupported Mac and Python Combination</source>
        <translation>不支援的 Mac 與 Python 組合</translation>
    </message>
    <message>
        <source>Up to</source>
        <translation>最多</translation>
    </message>
    <message>
        <source>Writing the file...</source>
        <translation>正在寫入檔案...</translation>
    </message>
    <message>
        <source>You place the corners, the same as on any QGIS layer.</source>
        <translation>由您自行放置角點，與在任何 QGIS 圖層中操作相同。</translation>
    </message>
    <message>
        <source>Your examples drive the search - naming the object makes it even more accurate.</source>
        <translation>您的範例將引導搜尋，為物件命名可讓結果更精確。</translation>
    </message>
    <message>
        <source>Your free detections come back on {date}.</source>
        <translation>您的免費偵測次數將於 {date} 恢復。</translation>
    </message>
    <message>
        <source>Your network proxy requires a username and password. Enter them in QGIS &gt; Settings &gt; Options &gt; Network, then restart QGIS and try again.</source>
        <translation>您的網路代理伺服器需要使用者名稱與密碼。請在 QGIS &gt; 偏好設定 &gt; 選項 &gt; 網路 中輸入，然後重新啟動 QGIS 並再試一次。</translation>
    </message>
    <message>
        <source>a month ago</source>
        <translation>一個月前</translation>
    </message>
    <message>
        <source>a week ago</source>
        <translation>一週前</translation>
    </message>
    <message>
        <source>a year ago</source>
        <translation>一年前</translation>
    </message>
    <message>
        <source>applies to every polygon</source>
        <translation>套用至所有多邊形</translation>
    </message>
    <message>
        <source>fine-tune the edges</source>
        <translation>微調邊緣</translation>
    </message>
    <message>
        <source>hide anything outside this range</source>
        <translation>隱藏此範圍外的所有內容</translation>
    </message>
    <message>
        <source>how each outline is styled</source>
        <translation>每條輪廓的樣式</translation>
    </message>
    <message>
        <source>how the outline is styled</source>
        <translation>輪廓的樣式</translation>
    </message>
    <message>
        <source>{count} polygons added so far.</source>
        <translation>目前已新增 {count} 個多邊形。</translation>
    </message>
    <message>
        <source>{count} vertices</source>
        <translation>{count} 個頂點</translation>
    </message>
    <message>
        <source>{credits} credits</source>
        <translation>{credits} 點數</translation>
    </message>
    <message>
        <source>{hidden} hidden by the filters</source>
        <translation>{hidden} 個因篩選條件而隱藏</translation>
    </message>
    <message>
        <source>{n} &quot;{object}&quot; found so far</source>
        <translation>目前已找到 {n} 個「{object}」</translation>
    </message>
    <message>
        <source>{n} corrections this round</source>
        <translation>本輪已修正 {n} 次</translation>
    </message>
    <message>
        <source>{n} kept</source>
        <translation>已保留 {n} 個</translation>
    </message>
    <message>
        <source>{n} months ago</source>
        <translation>{n} 個月前</translation>
    </message>
    <message>
        <source>{n} objects</source>
        <translation>{n} 個物件</translation>
    </message>
    <message>
        <source>{n} point placed.</source>
        <translation>已放置 {n} 個點。</translation>
    </message>
    <message>
        <source>{n} points placed.</source>
        <translation>已放置 {n} 個點。</translation>
    </message>
    <message>
        <source>{n} results</source>
        <translation>{n} 筆結果</translation>
    </message>
    <message>
        <source>{n} shape(s) edited this session</source>
        <translation>本次工作階段已編輯 {n} 個形狀</translation>
    </message>
    <message>
        <source>{n} shapes picked. Press Enter to join.</source>
        <translation>已選取 {n} 個形狀，按 Enter 鍵即可合併。</translation>
    </message>
    <message>
        <source>{n} tile(s) took too long to load and are missing from this result.</source>
        <translation>{n} 個圖磚載入時間過長，未包含在此結果中。</translation>
    </message>
    <message>
        <source>{n} weeks ago</source>
        <translation>{n} 週前</translation>
    </message>
    <message>
        <source>{n} years ago</source>
        <translation>{n} 年前</translation>
    </message>
    <message>
        <source>{tiles} tiles</source>
        <translation>{tiles} 個圖磚</translation>
    </message>
    <!-- Correct step rework, install and precision wording (2026-07-30) -->
    <message>
        <source>5,000 credits a month.</source>
        <translation>每月 5,000 點數。</translation>
    </message>
    <message>
        <source>A free run covers up to {cap} credits. This one costs more.</source>
        <translation>免費執行最多涵蓋 {cap} 點數。此次執行需要更多點數。</translation>
    </message>
    <message>
        <source>Adding an object uses the free on-device AI, which is not installed yet. Install it now? It runs once and takes a few minutes. The review waits for it, then arms Add for you.</source>
        <translation>新增物件功能使用免費的本機 AI，但尚未安裝。現在安裝嗎？只需安裝一次，需要幾分鐘。安裝期間檢視畫面會等待，完成後「新增」會自動就緒。</translation>
    </message>
    <message>
        <source>At this precision {obj} is too small to spot - raise the precision.</source>
        <translation>在此精細度下，{obj}太小而難以偵測－請提高精細度。</translation>
    </message>
    <message>
        <source>Automatic mode scans your zone tile by tile. 1 tile = 1 credit, so this run costs about {n} credits. More precision splits the zone into more tiles, which costs more credits.</source>
        <translation>自動模式會逐一圖磚掃描您的範圍。1 個圖磚＝1 點數，因此此次執行約需 {n} 點數。精細度越高，範圍會切分成越多圖磚，耗費的點數也越多。</translation>
    </message>
    <message>
        <source>Cancel setup</source>
        <translation>取消設定</translation>
    </message>
    <message>
        <source>Clean up this outline</source>
        <translation>清理此輪廓</translation>
    </message>
    <message>
        <source>Click points around one look-alike, then double-click to close.</source>
        <translation>在一個相似物件周圍逐點點選，再按兩下以封閉。</translation>
    </message>
    <message>
        <source>Click points around one object, then double-click to close.</source>
        <translation>在一個物件周圍逐點點選，再按兩下以封閉。</translation>
    </message>
    <message>
        <source>Click the layer name to see it on the map</source>
        <translation>點擊圖層名稱即可在地圖上查看</translation>
    </message>
    <message>
        <source>Click to open your dashboard</source>
        <translation>點擊以開啟您的控制台</translation>
    </message>
    <message>
        <source>Connection is slow - still working, tiles already found are kept...</source>
        <translation>連線速度較慢－仍在處理中，已找到的圖磚會保留...</translation>
    </message>
    <message>
        <source>Could not save your detections to a file.</source>
        <translation>無法將偵測結果儲存至檔案。</translation>
    </message>
    <message>
        <source>Downloads Blocked by Your Network</source>
        <translation>下載已被您的網路封鎖</translation>
    </message>
    <message>
        <source>Draw another example</source>
        <translation>再繪製一個範例</translation>
    </message>
    <message>
        <source>Drawing (click to stop)</source>
        <translation>繪製中（點擊以停止）</translation>
    </message>
    <message>
        <source>Errors, versions and the words you type, linked to your account. Never your imagery, layers or coordinates.</source>
        <translation>錯誤、版本與您輸入的文字，會與您的帳戶建立關聯。絕不包含您的影像、圖層或座標。</translation>
    </message>
    <message>
        <source>Fixing a polygon uses the free on-device AI, which is not installed yet. Install it now? It runs once and takes a few minutes. The review waits for it, then opens this polygon for you.</source>
        <translation>修正多邊形功能使用免費的本機 AI，但尚未安裝。現在安裝嗎？只需安裝一次，需要幾分鐘。安裝期間檢視畫面會等待，完成後會自動開啟此多邊形。</translation>
    </message>
    <message>
        <source>Fixing needs a one-time setup</source>
        <translation>修正需要進行一次性設定</translation>
    </message>
    <message>
        <source>Helps us fix bugs faster.</source>
        <translation>協助我們更快修復錯誤。</translation>
    </message>
    <message>
        <source>Higher precision splits the zone into more tiles. Each tile costs 1 credit and captures smaller objects.</source>
        <translation>精細度越高，範圍會被切分成更多圖磚。每個圖磚耗費 1 點數，並可捕捉更小的物件。</translation>
    </message>
    <message>
        <source>Keep detecting without limits:</source>
        <translation>無限制持續偵測：</translation>
    </message>
    <message>
        <source>Keep installing</source>
        <translation>繼續安裝</translation>
    </message>
    <message>
        <source>Keep this shape. The polygon stays picked, so you can still adjust, merge or delete it.</source>
        <translation>保留此形狀。多邊形會維持選取狀態，您仍可調整、合併或刪除它。</translation>
    </message>
    <message>
        <source>Loading the imagery...</source>
        <translation>正在載入影像...</translation>
    </message>
    <message>
        <source>Loading the imagery... {n}s</source>
        <translation>正在載入影像...{n} 秒</translation>
    </message>
    <message>
        <source>Max precision for this zone - draw a larger zone to go finer.</source>
        <translation>此範圍已達最高精細度－請繪製較大的範圍以進一步提高精細度。</translation>
    </message>
    <message>
        <source>More precision cuts the zone into more tiles and costs more credits.</source>
        <translation>精細度越高，範圍會切分成越多圖磚，耗費的點數也越多。</translation>
    </message>
    <message>
        <source>More precision finds smaller objects.</source>
        <translation>更高的精細度可偵測更小的物件。</translation>
    </message>
    <message>
        <source>More precision keeps helping {obj} in this zone.</source>
        <translation>在此範圍中，提高精細度仍能持續改善{obj}的偵測效果。</translation>
    </message>
    <message>
        <source>Name the object (or draw an example) first - Precision then tunes itself to it.</source>
        <translation>請先命名物件（或繪製範例），精細度會隨之自動調整。</translation>
    </message>
    <message>
        <source>Next: clean up the outlines</source>
        <translation>下一步：清理輪廓</translation>
    </message>
    <message>
        <source>Next: fix what looks wrong</source>
        <translation>下一步：修正看起來不對的地方</translation>
    </message>
    <message>
        <source>No detection in this zone. Try a more specific object word, or more precision.</source>
        <translation>此範圍內沒有偵測結果，請嘗試更具體的物件名稱，或提高精細度。</translation>
    </message>
    <message>
        <source>No image over this zone at this precision, so nothing was analyzed (not charged). Lower Precision, or pick a layer that covers this area.</source>
        <translation>此精細度下，此範圍內沒有影像，因此未進行分析（不計費）。請降低精細度，或選擇涵蓋此區域的圖層。</translation>
    </message>
    <message>
        <source>Not enough credits: {n} tiles, only {left} left. Lower the precision or the zone.</source>
        <translation>點數不足：需要 {n} 個圖磚，僅剩 {left} 個。請降低精細度或縮小範圍。</translation>
    </message>
    <message>
        <source>Opens your terra-lab.ai dashboard: your plan, your credits and your payment details.</source>
        <translation>開啟您的 terra-lab.ai 控制台：您的方案、點數與付款資訊。</translation>
    </message>
    <message>
        <source>Outline settings</source>
        <translation>輪廓設定</translation>
    </message>
    <message>
        <source>Precision</source>
        <translation>精細度</translation>
    </message>
    <message>
        <source>Ready for Automatic mode</source>
        <translation>自動模式已就緒</translation>
    </message>
    <message>
        <source>Running low: {n} free detections left, back on {date}. &lt;a href=&quot;{url}&quot;&gt;Upgrade to Pro&lt;/a&gt; to keep going.</source>
        <translation>免費次數即將用完：剩餘 {n} 次免費偵測，{date} 恢復。&lt;a href=&quot;{url}&quot;&gt;升級至 Pro&lt;/a&gt;即可繼續使用。</translation>
    </message>
    <message>
        <source>Running low: {n} free detections left. &lt;a href=&quot;{url}&quot;&gt;Upgrade to Pro&lt;/a&gt; to keep going.</source>
        <translation>額度即將用盡：剩餘 {n} 次免費偵測。&lt;a href=&quot;{url}&quot;&gt;升級至 Pro&lt;/a&gt;以繼續使用。</translation>
    </message>
    <message>
        <source>Same setup as your last run - the result will match. Add an example or change the precision for a different result.</source>
        <translation>設定與您上次執行相同，結果會一致。新增範例或變更精細度即可得到不同的結果。</translation>
    </message>
    <message>
        <source>Setting up the on-device AI. This runs once and takes a few minutes. The review waits here until it is done.</source>
        <translation>正在設定本機 AI。此設定只需執行一次，需要幾分鐘。檢視畫面會在此等待，直到設定完成。</translation>
    </message>
    <message>
        <source>Setting up the on-device AI...</source>
        <translation>正在設定本機 AI...</translation>
    </message>
    <message>
        <source>Shadows getting detected instead of trees? Use 'Exclude a look-alike' on one shadow - the AI drops similar false positives.</source>
        <translation>偵測到陰影而非樹木？在一處陰影上使用「排除相似物件」，AI 就會捨棄類似的誤判結果。</translation>
    </message>
    <message>
        <source>Share of each outline's points to keep. 100% is the class default.
Lower thins the smallest detail first, keeps the corners, and gives Right angles straight walls to square.</source>
        <translation>每條輪廓要保留的節點比例。100% 為類別預設值。
調低會優先精簡最細微的細節，保留角點，並為「直角化」提供更平直的牆面依據。</translation>
    </message>
    <message>
        <source>Share of the outline's points to keep. 100% is the standard density.
Lower thins the smallest detail first, keeps the corners, and gives Right angles straight walls to square.</source>
        <translation>輪廓要保留的節點比例。100% 為標準密度。
調低會優先精簡最細微的細節，保留角點，並為「直角化」提供更平直的牆面依據。</translation>
    </message>
    <message>
        <source>Share usage statistics with TerraLab</source>
        <translation>與 TerraLab 分享使用統計資料</translation>
    </message>
    <message>
        <source>Simplify this outline first</source>
        <translation>請先精簡此輪廓</translation>
    </message>
    <message>
        <source>Stop the setup</source>
        <translation>停止設定</translation>
    </message>
    <message>
        <source>Stop the setup and go back to the review. The AI fix stays unavailable until you install it.</source>
        <translation>停止設定並返回檢視畫面。AI 修正功能在您安裝前將持續無法使用。</translation>
    </message>
    <message>
        <source>Stop the setup?</source>
        <translation>停止設定？</translation>
    </message>
    <message>
        <source>The detection stopped responding before any tile came back. Check your connection, then run Detect again (nothing was charged).</source>
        <translation>偵測已停止回應，且尚未傳回任何圖磚。請檢查您的網路連線，然後重新執行偵測（不計費）。</translation>
    </message>
    <message>
        <source>The file may be open in QGIS or in another program. Close it and try Finish again.</source>
        <translation>此檔案可能已在 QGIS 或其他程式中開啟。請將其關閉，然後再次點擊「完成」。</translation>
    </message>
    <message>
        <source>The imagery is loading slowly... {n}s</source>
        <translation>影像載入速度較慢...{n} 秒</translation>
    </message>
    <message>
        <source>The on-device AI could not start, so the AI fix is off. Your detections are safe: switch the fix method to Manual to keep correcting, or save them as they are.</source>
        <translation>本機 AI 無法啟動，因此 AI 修正功能已停用。您的偵測結果仍安全：可將修正方式切換為手動以繼續修正，或直接以目前狀態儲存。</translation>
    </message>
    <message>
        <source>The on-device AI is unavailable, so the AI fix is off. Switch the fix method to Manual to keep correcting.</source>
        <translation>本機 AI 無法使用，因此 AI 修正功能已停用。請將修正方式切換為手動以繼續修正。</translation>
    </message>
    <message>
        <source>The on-device AI will not be installed, so fixing a polygon with it stays unavailable. What is already downloaded is kept, so starting again resumes from there.</source>
        <translation>本機 AI 將不會安裝，因此使用它修正多邊形的功能仍無法使用。已下載的內容會保留，重新開始時會從該處繼續。</translation>
    </message>
    <message>
        <source>The package index refused the download (error 403).

This is usually a company or campus network filtering downloads. Ask your IT administrator to allow pypi.org and files.pythonhosted.org, or run the install from another network.

Automatic (cloud) mode does not need this download.</source>
        <translation>套件索引拒絕了此次下載（錯誤 403）。

這通常是公司或校園網路過濾下載內容所致。請洽詢您的 IT 系統管理員，允許連線至 pypi.org 及 files.pythonhosted.org，或改用其他網路執行安裝。

自動（雲端）模式不需要此下載。</translation>
    </message>
    <message>
        <source>The raster file could not be found:
{path}

It may have been moved or renamed, or the drive or network share it is on may be disconnected. Reload the layer from where the file is now, then start again.</source>
        <translation>找不到點陣圖檔案：
{path}

它可能已被移動或重新命名，或是所在的磁碟機或網路共用資料夾已中斷連線。請從檔案目前的位置重新載入圖層，然後重新開始。</translation>
    </message>
    <message>
        <source>Thin this outline before you drag its corners, without moving the dials that drive the whole layer.</source>
        <translation>在拖曳角點前先精簡此輪廓，且不影響驅動整個圖層的控制項。</translation>
    </message>
    <message>
        <source>This area is large for this precision. Raise the precision or zoom in for sharper detections.</source>
        <translation>對此精細度而言，此範圍偏大。請提高精細度或放大以取得更清晰的偵測結果。</translation>
    </message>
    <message>
        <source>This example is very small at this precision. Raise the precision or draw a larger object.</source>
        <translation>此範例在此精細度下非常小。請提高精細度或繪製更大的物件。</translation>
    </message>
    <message>
        <source>This example is very small even at full precision. Draw a larger object, or it may be too small to detect.</source>
        <translation>即使在最高精細度下，此範例仍非常小。請繪製更大的物件，否則可能太小而無法偵測。</translation>
    </message>
    <message>
        <source>This layer has no image over your zone at this precision. The map source answered with an empty tile, so there is nothing to detect on. Lower Precision, zoom the layer out until the imagery shows, or pick a layer that covers this area.</source>
        <translation>此圖層在此精細度下，您的範圍內沒有影像。地圖來源回傳了空白圖磚，因此沒有可供偵測的內容。請降低精細度、縮小圖層直到顯示影像，或選擇涵蓋此區域的圖層。</translation>
    </message>
    <message>
        <source>This polygon only. Every other one follows the Shapes step.</source>
        <translation>僅此多邊形。其他所有多邊形皆依照「形狀」步驟設定。</translation>
    </message>
    <message>
        <source>This polygon only. Fewer points means fewer corners to drag.</source>
        <translation>僅此多邊形。節點越少，可拖曳的角點也越少。</translation>
    </message>
    <message>
        <source>This run costs more credits than a free run covers.</source>
        <translation>此次執行所需點數超過免費額度。</translation>
    </message>
    <message>
        <source>This zone at this precision needs a subscription. Lower the precision or the zone to stay free.</source>
        <translation>此精細度下的此範圍需要訂閱。請降低精細度或縮小範圍以維持免費。</translation>
    </message>
    <message>
        <source>This zone is too large for sharp detections, even at full precision. Draw a smaller zone for the best results.</source>
        <translation>即使在最高精細度下，此範圍仍過大而無法取得清晰的偵測結果。請繪製較小的範圍以取得最佳效果。</translation>
    </message>
    <message>
        <source>Tiles your layer has no image for are dropped before they are sent, so a run can cost less than this, never more.</source>
        <translation>您的圖層沒有影像涵蓋的圖磚，會在傳送前被捨棄，因此實際花費可能低於此金額，但絕不會更高。</translation>
    </message>
    <message>
        <source>Type a word for the object, or draw an example.</source>
        <translation>輸入物件的名稱，或繪製範例。</translation>
    </message>
    <message>
        <source>Upgrade to Pro to finish this zone: 5,000 credits/month.</source>
        <translation>升級至 Pro 以完成此範圍：每月 5,000 點數。</translation>
    </message>
    <message>
        <source>Wait for the on-device AI to finish installing.</source>
        <translation>請等待本機 AI 完成安裝。</translation>
    </message>
    <message>
        <source>{n} {object} saved</source>
        <translation>已儲存 {n} 個 {object}</translation>
    </message>
    <message>
        <source>{n} {object} saved to {layer}</source>
        <translation>已儲存 {n} 個 {object} 至 {layer}</translation>
    </message>
    <!-- Correct-step on-device AI setup, free-run cap wording, missing raster (2026-07-30) -->
    <message>
        <source>5,000 objects every month</source>
        <translation>每月 5,000 個物件</translation>
    </message>
    <message>
        <source>A session is already running.</source>
        <translation>工作階段已在執行中。</translation>
    </message>
    <message>
        <source>AI engine</source>
        <translation>AI 引擎</translation>
    </message>
    <message>
        <source>AI fixing is not reachable right now. Switched to editing by hand, which works offline.</source>
        <translation>AI 修正功能目前無法連線，已切換為手動編輯，可離線使用。</translation>
    </message>
    <message>
        <source>AI ready</source>
        <translation>AI 已就緒</translation>
    </message>
    <message>
        <source>Add an object the AI missed. In AI, point at it and the model outlines it, free; in Manual, draw its corners.</source>
        <translation>新增 AI 遺漏的物件：AI 方式下，點一下即可由模型免費勾勒輪廓；手動方式下，自行繪製其角點。</translation>
    </message>
    <message>
        <source>Almost ready: the AI file did not download.</source>
        <translation>即將就緒：AI 檔案未下載成功。</translation>
    </message>
    <message>
        <source>Almost ready: the AI file is still missing.</source>
        <translation>即將就緒：仍缺少 AI 檔案。</translation>
    </message>
    <message>
        <source>Answered on your computer this time. TerraLab could not be reached.</source>
        <translation>這次已在您的電腦上完成回應。TerraLab 目前無法連線。</translation>
    </message>
    <message>
        <source>As fine as {obj} benefits from - finer splits them into pieces.</source>
        <translation>已達到{obj}能受益的最高精細度－再更精細會將其拆成碎片。</translation>
    </message>
    <message>
        <source>Automatic mode is ready. The on-device AI could not be installed, so Semi-Auto mode and the AI fix are off until it is. Everything else works.</source>
        <translation>自動模式已就緒。本機 AI 無法安裝，因此半自動模式與 AI 修正功能在安裝完成前無法使用。其餘功能皆正常運作。</translation>
    </message>
    <message>
        <source>Automatic mode needs a small one-time setup before it can read your imagery. It takes about a minute.</source>
        <translation>自動模式需要先進行簡單的一次性設定，才能讀取您的影像。大約需要一分鐘。</translation>
    </message>
    <message>
        <source>Automatic mode ready</source>
        <translation>自動模式已就緒</translation>
    </message>
    <message>
        <source>Categories</source>
        <translation>類別</translation>
    </message>
    <message>
        <source>Choose between Semi-Auto and Automatic segmentation</source>
        <translation>選擇半自動或自動分割方式</translation>
    </message>
    <message>
        <source>Choose where the AI runs: on TerraLab servers, or on your own computer</source>
        <translation>選擇 AI 的執行位置：在 TerraLab 伺服器上，或在您自己的電腦上</translation>
    </message>
    <message>
        <source>Click the object you want to segment:</source>
        <translation>點擊您要分割的物件：</translation>
    </message>
    <message>
        <source>Cloud AI</source>
        <translation>Cloud AI</translation>
    </message>
    <message>
        <source>Cloud AI needs your account, and it is signed out. Sign back in, or install the offline AI to work without one.</source>
        <translation>Cloud AI 需要您的帳戶，但目前已登出。請重新登入，或安裝本機 AI，不需帳戶即可使用。</translation>
    </message>
    <message>
        <source>How Cloud AI works</source>
        <translation>Cloud AI 的運作方式</translation>
    </message>
    <message>
        <source>Continue with Cloud AI</source>
        <translation>繼續使用 Cloud AI</translation>
    </message>
    <message>
        <source>Delete this polygon and leave the manual edit. Anything you changed here and did not save goes with it. Undo brings the polygon back.</source>
        <translation>刪除此多邊形並離開手動編輯。您在此處所做但尚未儲存的變更都會一併遺失。復原可將多邊形還原。</translation>
    </message>
    <message>
        <source>Downloading and setting it up takes &lt;b&gt;about {n} minutes&lt;/b&gt;, once.</source>
        <translation>下載並設定大約需要 &lt;b&gt;{n} 分鐘&lt;/b&gt;，僅需一次。</translation>
    </message>
    <message>
        <source>Draw a zone, name one kind of object, and get all of them in one run. Use Semi-Auto mode to work one object at a time.</source>
        <translation>繪製範圍、為一種物件命名，即可一次執行取得所有物件。使用半自動模式可逐一處理物件。</translation>
    </message>
    <message>
        <source>Each click sends a small square of the image to our servers in Europe, and the outline comes back.</source>
        <translation>每次點擊會將影像中的一小塊方形區域傳送到我們位於歐洲的伺服器，並回傳輪廓結果。</translation>
    </message>
    <message>
        <source>Ends this session. Your saved polygons are kept.</source>
        <translation>結束此工作階段，已儲存的多邊形會保留。</translation>
    </message>
    <message>
        <source>Faster and more accurate</source>
        <translation>更快、更準確</translation>
    </message>
    <message>
        <source>Featured</source>
        <translation>精選</translation>
    </message>
    <message>
        <source>Free, works offline</source>
        <translation>免費，離線可用</translation>
    </message>
    <message>
        <source>Install it now</source>
        <translation>立即安裝</translation>
    </message>
    <message>
        <source>Install the offline AI</source>
        <translation>安裝本機 AI</translation>
    </message>
    <message>
        <source>Installing the offline AI</source>
        <translation>正在安裝本機 AI</translation>
    </message>
    <message>
        <source>Intel Mac: using the older AI model.</source>
        <translation>Intel Mac：使用較舊版本的 AI 模型。</translation>
    </message>
    <message>
        <source>It needs &lt;b&gt;{gb} GB&lt;/b&gt; of free disk space.</source>
        <translation>需要 &lt;b&gt;{gb} GB&lt;/b&gt; 的可用磁碟空間。</translation>
    </message>
    <message>
        <source>My computer</source>
        <translation>我的電腦</translation>
    </message>
    <message>
        <source>My computer, free and unlimited</source>
        <translation>我的電腦，免費且無使用限制</translation>
    </message>
    <message>
        <source>My work</source>
        <translation>我的作品</translation>
    </message>
    <message>
        <source>No credits left. Each object you save with Cloud AI costs one credit.</source>
        <translation>點數已用完。透過 Cloud AI 儲存的每個物件都需要花費 1 點數。</translation>
    </message>
    <message>
        <source>No credits left. Each object you save with Cloud AI costs one credit. Switch to the offline AI to keep working for free, or upgrade from the panel.</source>
        <translation>點數已用完。透過 Cloud AI 儲存的每個物件都需要花費 1 點數。切換至本機 AI 即可繼續免費使用，或從面板升級方案。</translation>
    </message>
    <message>
        <source>No credits left. This polygon stays on the map, and Export still works.</source>
        <translation>點數已用完。此多邊形會留在地圖上，Export 功能仍可使用。</translation>
    </message>
    <message>
        <source>No credits left. This polygon stays on the map, but it cannot be saved.</source>
        <translation>點數已用完。此多邊形會留在地圖上，但無法儲存。</translation>
    </message>
    <message>
        <source>No object found here. Try clicking somewhere else.</source>
        <translation>此處未找到物件。請嘗試點擊其他位置。</translation>
    </message>
    <message>
        <source>Not now</source>
        <translation>先不要</translation>
    </message>
    <message>
        <source>Your project and your files stay on your computer. One credit per object you save.</source>
        <translation>您的專案與檔案都留在您的電腦上。每儲存 1 個物件需要 1 點數。</translation>
    </message>
    <message>
        <source>One precision level fits {obj} in a zone this size - draw a larger zone for a choice.</source>
        <translation>此範圍大小下，只有一種精細度適合{obj}－請繪製較大的範圍以取得更多選擇。</translation>
    </message>
    <message>
        <source>One-time setup</source>
        <translation>一次性設定</translation>
    </message>
    <message>
        <source>Other detections</source>
        <translation>其他偵測結果</translation>
    </message>
    <message>
        <source>Pick a raster layer and accept the Terms to start.</source>
        <translation>選擇一個點陣圖圖層並接受服務條款，即可開始使用。</translation>
    </message>
    <message>
        <source>Preparing the install...</source>
        <translation>正在準備安裝...</translation>
    </message>
    <message>
        <source>Pro is active on this account. Your credits are ready.</source>
        <translation>此帳戶的 Pro 方案已生效。您的點數已就緒。</translation>
    </message>
    <message>
        <source>Read the privacy policy</source>
        <translation>閱讀隱私政策</translation>
    </message>
    <message>
        <source>Right-click a polygon on the map to delete it.</source>
        <translation>在地圖上右鍵點擊多邊形即可刪除。</translation>
    </message>
    <message>
        <source>Running low: {n} credits left, back on {date}. &lt;a href=&quot;{url}&quot;&gt;Upgrade to Pro&lt;/a&gt; to keep going.</source>
        <translation>點數即將用盡：剩餘 {n} 點，{date} 恢復。&lt;a href=&quot;{url}&quot;&gt;升級至 Pro&lt;/a&gt;即可繼續使用。</translation>
    </message>
    <message>
        <source>Running low: {n} credits left. &lt;a href=&quot;{url}&quot;&gt;Upgrade to Pro&lt;/a&gt; to keep going.</source>
        <translation>點數即將用盡：剩餘 {n} 點。&lt;a href=&quot;{url}&quot;&gt;升級至 Pro&lt;/a&gt;以繼續使用。</translation>
    </message>
    <message>
        <source>Save a polygon first. Export writes every polygon you kept to a layer.</source>
        <translation>請先儲存一個多邊形。Export 會將您保留的每個多邊形寫入圖層。</translation>
    </message>
    <message>
        <source>Saving object</source>
        <translation>正在儲存物件</translation>
    </message>
    <message>
        <source>Semi-Auto</source>
        <translation>半自動</translation>
    </message>
    <message>
        <source>Semi-Auto mode installs local components that are not available for this Mac with this version of QGIS. Please use Automatic mode instead, which runs fully in the cloud and needs no local install.</source>
        <translation>半自動模式需要安裝本機元件，但在此 Mac 與此版本的 QGIS 下不支援。請改用自動模式，它完全在雲端執行，不需要本機安裝。</translation>
    </message>
    <message>
        <source>Semi-Auto mode is not supported</source>
        <translation>不支援半自動模式</translation>
    </message>
    <message>
        <source>Semi-Auto mode is not supported in this QGIS installation</source>
        <translation>此 QGIS 安裝環境不支援半自動模式</translation>
    </message>
    <message>
        <source>Semi-Auto mode needs to install local dependencies, which is not supported inside this sandboxed QGIS installation (Flatpak or Snap). Please use Automatic mode instead, which runs fully in the cloud and needs no local install.</source>
        <translation>半自動模式需要安裝本機相依套件，但在此沙盒化的 QGIS 環境（Flatpak 或 Snap）中不支援。請改用自動模式，它完全在雲端執行，不需要本機安裝。</translation>
    </message>
    <message>
        <source>Semi-Auto mode stays free and unlimited on your computer.</source>
        <translation>半自動模式在您的電腦上永久免費且無使用限制。</translation>
    </message>
    <message>
        <source>Session ended</source>
        <translation>工作階段已結束</translation>
    </message>
    <message>
        <source>Set up now</source>
        <translation>立即設定</translation>
    </message>
    <message>
        <source>Setting up on your computer {dot} &lt;b&gt;pick Cloud AI to start now&lt;/b&gt;</source>
        <translation>正在您的電腦上設定 {dot} &lt;b&gt;選擇 Cloud AI 立即開始&lt;/b&gt;</translation>
    </message>
    <message>
        <source>Start Semi-Auto AI Segmentation</source>
        <translation>開始半自動 AI Segmentation</translation>
    </message>
    <message>
        <source>Stop and use my computer instead</source>
        <translation>停止並改用我的電腦</translation>
    </message>
    <message>
        <source>Stop the install</source>
        <translation>停止安裝</translation>
    </message>
    <message>
        <source>The AI is still loading. Try again in a few seconds.</source>
        <translation>AI 仍在載入中，請稍後幾秒再試一次。</translation>
    </message>
    <message>
        <source>The AI was set up but could not start.</source>
        <translation>AI 已完成設定，但無法啟動。</translation>
    </message>
    <message>
        <source>The click was cancelled.</source>
        <translation>點擊已取消。</translation>
    </message>
    <message>
        <source>The install did not finish. Retry it, or pick Cloud AI.</source>
        <translation>安裝未完成。請重試，或選擇 Cloud AI。</translation>
    </message>
    <message>
        <source>The model is unsure about this outline. Click again to correct it, or draw it by hand.</source>
        <translation>模型對此輪廓沒有把握。請再次點擊以修正，或改為手動繪製。</translation>
    </message>
    <message>
        <source>The offline AI</source>
        <translation>本機 AI</translation>
    </message>
    <message>
        <source>The offline AI is not installed yet.</source>
        <translation>本機 AI 尚未安裝。</translation>
    </message>
    <message>
        <source>The offline AI is still downloading.</source>
        <translation>本機 AI 仍在下載中。</translation>
    </message>
    <message>
        <source>The smaller model on your computer. No credits, no limit.</source>
        <translation>在您電腦上執行的較小型模型。不需點數，沒有限制。</translation>
    </message>
    <message>
        <source>This deletes the local AI model files, signs you out, and resets the plugin. Your account and credits are not affected. Semi-Auto mode will download the files again next time you use it.</source>
        <translation>這會刪除本機的 AI 模型檔案、將您登出並重設外掛程式。您的帳戶與點數不受影響。下次使用半自動模式時會重新下載檔案。</translation>
    </message>
    <message>
        <source>This drive has {free} GB free, under the {need} GB the install needs. Free some space, or use Cloud AI.</source>
        <translation>此磁碟機有 {free} GB 可用空間，低於安裝所需的 {need} GB。請釋放一些空間，或使用 Cloud AI。</translation>
    </message>
    <message>
        <source>This image has no position on the map, so Automatic cannot place what it finds. Give it one with the QGIS Georeferencer, or use Semi-Auto mode on it as is.</source>
        <translation>此影像在地圖上沒有位置資訊，因此自動模式無法放置偵測結果。請使用 QGIS 地理參照工具為其設定位置，或直接以半自動模式處理。</translation>
    </message>
    <message>
        <source>This model rates every object the same, so filtering by confidence would show all of them or none. Use Size below, or fix objects in the next step.</source>
        <translation>此模型對每個物件的評分都相同，因此依信心度篩選只會顯示全部或完全不顯示。請改用下方的「尺寸」篩選，或在下一步修正物件。</translation>
    </message>
    <message>
        <source>This raster is rotated. Run Warp (Reproject) on it to straighten it first. Semi-Auto mode cannot read it either.</source>
        <translation>此點陣圖已旋轉。請先執行「Warp（重新投影）」將其校正。半自動模式同樣無法讀取此影像。</translation>
    </message>
    <message>
        <source>Use Cloud AI instead</source>
        <translation>改用 Cloud AI</translation>
    </message>
    <message>
        <source>Use my computer instead</source>
        <translation>改用我的電腦</translation>
    </message>
    <message>
        <source>Where the segmentation runs:</source>
        <translation>分割作業的執行位置：</translation>
    </message>
    <message>
        <source>You can close this window and keep working. The install carries on, and the panel shows how far it has got.</source>
        <translation>您可以關閉此視窗並繼續工作。安裝會繼續進行，面板會顯示目前的進度。</translation>
    </message>
    <message>
        <source>Your free credits come back on {date}.</source>
        <translation>您的免費點數將於 {date} 恢復。</translation>
    </message>
    <message>
        <source>Everything stays on this computer {dot} &lt;b&gt;about 10 minutes to install&lt;/b&gt;</source>
        <translation>一切都留在這台電腦上 {dot} &lt;b&gt;安裝約需 10 分鐘&lt;/b&gt;</translation>
    </message>
    <message>
        <source>Everything stays on this computer {dot} &lt;b&gt;save as many as you like&lt;/b&gt;</source>
        <translation>一切都留在這台電腦上 {dot} &lt;b&gt;儲存物件不限量&lt;/b&gt;</translation>
    </message>
    <message>
        <source>Everything stays on this computer {dot} &lt;b&gt;{gb} GB and about 10 minutes to install&lt;/b&gt;</source>
        <translation>一切都留在這台電腦上 {dot} &lt;b&gt;{gb} GB，安裝約需 10 分鐘&lt;/b&gt;</translation>
    </message>
    <message>
        <source>No setup on this computer {dot} &lt;b&gt;1 credit per object you save&lt;/b&gt;</source>
        <translation>這台電腦不需設定 {dot} &lt;b&gt;每儲存 1 個物件 1 點數&lt;/b&gt;</translation>
    </message>
    <message>
        <source>The offline AI answers your clicks on this computer. Your imagery stays here, and every click is free.</source>
        <translation>本機 AI 會在這台電腦上回應您的點擊。您的影像留在這裡，每次點擊都免費。</translation>
    </message>
    <message>
        <source>This online layer returned no imagery for this area. Its server refused the request. Check the layer&apos;s URL in Layer Properties, or use another basemap.</source>
        <translation>此線上圖層沒有為此區域回傳任何影像。其伺服器拒絕了這次請求。請在圖層屬性中檢查該圖層的網址，或使用其他底圖。</translation>
    </message>
    <message>
        <source>1 credit covers about 0.17 km² at default precision.</source>
        <translation>在預設精細度下，1 點數約可涵蓋 0.17 平方公里。</translation>
    </message>
    <message>
        <source>Clear all {n}</source>
        <translation>全部清除 {n}</translation>
    </message>
    <message>
        <source>Could not check your AI Segmentation account. If this lasts, sign out and sign in again.</source>
        <translation>無法檢查您的 AI Segmentation 帳戶。如果問題持續發生，請登出後再重新登入。</translation>
    </message>
    <message>
        <source>Could not load your account. Try again in a moment.</source>
        <translation>無法載入您的帳戶。請稍後再試一次。</translation>
    </message>
    <message>
        <source>Could not reach TerraLab. Check your internet connection, then try again.</source>
        <translation>無法連線至 TerraLab。請檢查您的網路連線，然後再試一次。</translation>
    </message>
    <message>
        <source>Editing by hand could not open on this polygon on its own. Try again, or fix it with the AI.</source>
        <translation>手動編輯無法自動在此多邊形上開啟。請再試一次，或改用 AI 修正。</translation>
    </message>
    <message>
        <source>It downloads first, and takes about 10 minutes.</source>
        <translation>會先下載，大約需要 10 分鐘。</translation>
    </message>
    <message>
        <source>It downloads first: {gb} GB and about 10 minutes.</source>
        <translation>會先下載：{gb} GB，大約需要 10 分鐘。</translation>
    </message>
    <message>
        <source>Nothing in this category yet.</source>
        <translation>此類別目前尚無內容。</translation>
    </message>
    <message>
        <source>One run covers up to {cap} tiles. This zone at this precision needs more. Draw a smaller zone, or lower the precision.</source>
        <translation>一次執行最多涵蓋 {cap} 個圖磚。此範圍在此精細度下需要更多圖磚。請縮小範圍，或降低精細度。</translation>
    </message>
    <message>
        <source>QGIS could not open a browser. Open this address to finish signing in, then come back here. It works once:
{}</source>
        <translation>QGIS 無法開啟瀏覽器。請開啟此網址以完成登入，然後返回這裡。僅能使用一次：
{}</translation>
    </message>
    <message>
        <source>QGIS could not open a browser. The sign-in address is copied to your clipboard: paste it into a browser to finish, then come back here. It works once.</source>
        <translation>QGIS 無法開啟瀏覽器。登入網址已複製到您的剪貼簿：請將其貼到瀏覽器中以完成登入，然後返回這裡。僅能使用一次。</translation>
    </message>
    <message>
        <source>Reading your logs...</source>
        <translation>正在讀取您的日誌...</translation>
    </message>
    <message>
        <source>Sign in again to fix with the AI. Switched to editing by hand, which needs no account.</source>
        <translation>請重新登入以使用 AI 修正。已切換為手動編輯，不需要帳戶。</translation>
    </message>
    <message>
        <source>The export did not finish. Your polygons are still on the map, so you can try again.</source>
        <translation>Export 未完成。您的多邊形仍留在地圖上，因此您可以再試一次。</translation>
    </message>
    <message>
        <source>The install did not finish {dot} &lt;b&gt;retry it, or pick Cloud AI&lt;/b&gt;</source>
        <translation>安裝未完成 {dot} &lt;b&gt;重試，或選擇 Cloud AI&lt;/b&gt;</translation>
    </message>
    <message>
        <source>The polygons could not be put into the new layer, so nothing was saved. They are still on the map, so you can try again.</source>
        <translation>多邊形無法加入新圖層，因此未儲存任何內容。它們仍留在地圖上，因此您可以再試一次。</translation>
    </message>
    <message>
        <source>This layer has no usable position on the map. Open it in QGIS and check its extent.</source>
        <translation>此圖層在地圖上沒有可用的位置資訊。請在 QGIS 中開啟它並檢查其範圍。</translation>
    </message>
    <message>
        <source>This run found nothing. Add the object yourself below, or press Exit and run again with another word or a smaller zone.</source>
        <translation>此次執行未找到任何物件。請在下方自行新增物件，或按下「結束」，使用其他文字或縮小範圍後重新執行。</translation>
    </message>
    <message>
        <source>This zone at this precision needs more tiles than one run covers. Draw a smaller zone, or lower the precision.</source>
        <translation>此範圍在此精細度下，所需的圖磚數量超過一次執行可涵蓋的上限。請縮小範圍，或降低精細度。</translation>
    </message>
    <message>
        <source>Undo every correction of this round at once. The count is in the label, so you can see what goes.</source>
        <translation>一次復原本輪的所有修正。數量顯示在標籤上，方便您掌握會復原多少項。</translation>
    </message>
    <message>
        <source>You are out of credits, so the AI fix cannot answer. Switched to editing by hand, which is free.</source>
        <translation>您的點數已用完，因此 AI 修正功能無法回應。已切換為手動編輯，可免費使用。</translation>
    </message>
    <message>
        <source>You are signed in on this computer, but QGIS cannot read your sign-in until you enter its master password.</source>
        <translation>您已在這台電腦上登入，但在您輸入 QGIS 的主密碼之前，QGIS 無法讀取您的登入資訊。</translation>
    </message>
</context>
</TS>
