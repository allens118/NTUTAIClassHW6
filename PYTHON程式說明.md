# Python 程式說明

本文件整理本專案中較重要的 Python 檔案，以及其中關鍵函式的用途與角色，方便閱讀程式架構、撰寫報告或進行口頭說明。

## 1. 重要 Python 檔案

### `train_diffusion.py`
這是本專案最核心的主程式，負責完成整個擴散模型訓練流程，包含：

- 資料集讀取與前處理
- U-Net 模型建立
- diffusion schedule 建立
- 訓練與驗證
- 圖片生成與去噪視覺化
- loss curve 與結果摘要輸出

簡單來說，如果要執行模型訓練，主要就是執行這個檔案。

### `build_report_figures.py`
這個檔案的用途是根據已經訓練完成的結果，重新整理並輸出報告用圖表。

它主要負責：

- 讀取 `training_summary.json`
- 重新繪製較專業的 loss 圖
- 標註關鍵 epoch 與特殊點
- 建立 milestone 比較圖
- 建立不同 epoch 生成結果的整合比較圖

如果目的是「整理報告圖」，而不是重新訓練模型，則使用這個檔案較合適。

## 2. `train_diffusion.py` 重要類別與函式

### `set_seed(seed)`
用途：
設定 Python、NumPy 與 PyTorch 的亂數種子。

重要性：
這個函式可以讓每次訓練的隨機行為更穩定，提升實驗的可重現性。對於深度學習實驗來說，這是非常重要的基本設定。

### `PixivFacesDataset`
用途：
定義資料集類別，負責從 `crop_2020_img` 資料夾中讀取影像。

主要功能：

- 搜尋 `.jpg` 與 `.png` 影像
- 將影像轉成 RGB
- 調整成固定大小
- 轉成 tensor
- 做正規化

重要性：
這個類別是整個訓練流程的資料入口，沒有它就無法把資料送進模型中。

### `sinusoidal_time_embedding(timesteps, dim)`
用途：
把 diffusion 的時間步（timestep）轉換成可供神經網路使用的向量表示。

重要性：
擴散模型在不同 timestep 下要做的事情不同，因此模型必須知道目前正在處理哪一個時間步。這個函式提供了時間資訊的編碼方式。

### `ResidualBlock`
用途：
建立帶有殘差結構的卷積區塊，並將時間步嵌入資訊注入其中。

重要性：
這是 U-Net 的基本組成元件之一，能幫助模型更穩定地學習特徵，同時保留來自時間步的條件資訊。

### `DownBlock`
用途：
負責編碼器中的下採樣操作。

主要功能：

- 先做殘差特徵抽取
- 再透過卷積將特徵圖縮小
- 同時保留 skip connection 所需資訊

重要性：
讓模型能從較大尺寸的影像逐步萃取出更高層次的特徵。

### `UpBlock`
用途：
負責解碼器中的上採樣操作。

主要功能：

- 將低解析度特徵放大
- 與對應的 skip connection 特徵進行串接
- 再做殘差特徵整合

重要性：
幫助模型從壓縮過的特徵中逐步恢復空間結構，對生成影像品質非常重要。

### `SimpleUNet`
用途：
建立整個 DDPM 使用的 U-Net 主幹網路。

主要功能：

- 接收帶噪影像
- 接收時間步資訊
- 預測該 timestep 下的噪聲

重要性：
這是整個模型最核心的神經網路，因為 DDPM 的訓練目標就是讓模型學會「預測噪聲」。

### `DiffusionSchedule`
用途：
建立並管理擴散模型在前向與反向過程中需要用到的各種係數。

主要內容包括：

- `betas`
- `alphas`
- `alpha_bars`
- posterior variance

重要性：
這個類別是擴散模型數學機制的核心，負責把公式中的參數實際整理成可供訓練與取樣使用的形式。

#### `extract(...)`
用途：
依照當前 timestep 從預先計算好的係數中取出對應值。

重要性：
讓不同樣本在不同時間步時，都能正確使用對應的 diffusion 參數。

#### `q_sample(x_start, timesteps, noise)`
用途：
執行前向擴散，也就是對乾淨影像加上噪聲。

重要性：
這是生成訓練資料 `x_t` 的方式，模型就是透過學習如何從 `x_t` 預測噪聲來完成訓練。

#### `predict_x0(x_t, timesteps, pred_noise)`
用途：
根據目前帶噪影像與預測噪聲，回推出原始乾淨影像的估計值。

重要性：
這個函式有助於理解模型在每個 timestep 下如何回復原始影像資訊。

#### `p_sample(model, x, timesteps)`
用途：
執行一次反向去噪步驟。

重要性：
這是影像生成時最重要的核心步驟之一，模型會逐步把隨機噪聲轉成有意義的影像。

#### `sample(...)`
用途：
從純隨機噪聲開始，反覆執行反向去噪，直到生成完整影像。

重要性：
這個函式就是最後產生新圖片的地方，也是報告中生成結果圖片的來源。

### `denormalize(images)`
用途：
把已經正規化到 `[-1, 1]` 的影像還原回 `[0, 1]`。

重要性：
模型訓練時使用的是正規化影像，但在輸出圖片與視覺化時，必須還原成正常顯示範圍。

### `save_loss_curve(train_losses, val_losses, path)`
用途：
繪製訓練損失與驗證損失曲線。

重要性：
這是觀察模型是否收斂、是否過擬合的重要圖表來源。

### `save_curated_loss_curve(...)`
用途：
繪製較正式的報告版 loss curve，並只標註指定 epoch 與特殊點。

重要性：
這個函式比一般 loss curve 更適合放進報告，因為它強化了重點資訊的可讀性。

### `save_trajectory_grid(trajectory, path)`
用途：
將同一張影像在不同去噪階段的結果排列成圖。

重要性：
可以清楚展示 diffusion model 的反向過程，是報告中非常有代表性的視覺化結果。

### `save_sample_grid(samples, path, title)`
用途：
把多張生成圖片排成整齊的網格圖。

重要性：
這是展示模型生成效果最直接的方式之一。

### `save_epoch_comparison_figure(...)`
用途：
把多個不同 epoch 的樣本圖整理到同一張比較圖中。

重要性：
有助於觀察隨著訓練進行，生成結果是否逐步改善。

### `compute_style_metrics(dataset_samples, generated_samples)`
用途：
計算資料集樣本與生成樣本的簡易統計特徵。

包含指標：

- RGB 平均值
- RGB 標準差
- 飽和度
- 紋理強度

重要性：
這些指標雖然不是像 FID 那樣的標準生成評估指標，但仍能提供基本的風格差異參考。

### `evaluate_validation_loss(model, diffusion, dataloader, device)`
用途：
在驗證集上計算模型平均損失。

重要性：
驗證損失可以用來判斷模型在未見資料上的表現，也是選擇最佳 checkpoint 的重要依據。

### `create_data_loaders(config)`
用途：
建立訓練集與驗證集的 dataloader。

主要功能：

- 建立資料集
- 切分 train / validation
- 設定 batch size
- 設定 shuffle 與載入方式

重要性：
它是訓練資料流的控制中心。

### `train(config)`
用途：
執行完整訓練流程。

主要內容：

- 設定亂數種子
- 建立輸出資料夾
- 選擇裝置（CPU / GPU）
- 建立資料集與 dataloader
- 建立模型與 optimizer
- 執行每個 epoch 的訓練與驗證
- 儲存 sample、loss curve、checkpoint 與 summary

重要性：
這是整個專案最重要的主流程函式，也是所有功能整合的核心。

### `parse_args()`
用途：
讀取命令列參數。

可控制內容包括：

- 資料夾路徑
- 輸出路徑
- image size
- batch size
- epoch 數
- learning rate
- diffusion steps
- device
- save epochs

重要性：
讓同一份程式可以用不同設定重複執行，方便做不同實驗比較。

## 3. `build_report_figures.py` 重要函式

### `find_early_rebound_epoch(val_losses, end_epoch=30)`
用途：
在訓練前期找出 validation loss 回升最明顯的 epoch。

重要性：
有助於報告中標註 early rebound 這類「值得討論的特殊點」。

### `build_curated_loss_curve(output_dir, summary)`
用途：
根據 `training_summary.json` 重繪一張更適合報告的 loss 圖，並標出：

- 最佳 validation 點
- early rebound 點
- late rebound 點

重要性：
這是報告中最具分析價值的圖之一。

### `build_milestone_metrics(output_dir, summary)`
用途：
建立 milestone loss 與 generalization gap 的比較圖。

重要性：
能更清楚呈現不同 epoch 節點的表現差異，不只看單一曲線，而是直接比較特定里程碑。

### `build_epoch_comparison(output_dir)`
用途：
將多個關鍵 epoch 的生成結果合併成同一張圖。

重要性：
可作為報告中的視覺化主圖之一，直接展示不同訓練階段的生成品質差異。

### `main()`
用途：
作為 `build_report_figures.py` 的主執行入口。

主要流程：

- 讀取 `training_summary.json`
- 建立整理過的 loss 圖
- 建立 milestone 指標圖
- 建立 epoch 比較圖
- 更新 summary 內容

重要性：
讓整個報告圖重建流程可以單獨執行，不需要重新訓練模型。

## 4. 建議如何介紹這個專案

如果要簡短介紹，可以這樣說：

> 本專案使用 PyTorch 實作 DDPM 擴散模型，透過 U-Net 預測不同 timestep 下的噪聲，並在 Pixiv 臉部插畫資料集上進行訓練。除了模型訓練本身，也包含前處理、驗證、結果視覺化、特殊 epoch 分析與報告圖生成流程。

如果要再更口語一點，可以這樣說：

> `train_diffusion.py` 是核心訓練程式，`build_report_figures.py` 則負責把訓練結果整理成適合報告展示的圖表。
