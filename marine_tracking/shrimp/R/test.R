# 1. 安装与加载 -------------------------------------------------------------
library(rEDM)
# 2. 准备示例数据 -----------------------------------------------------------
# 这里使用内置的三物种 Lotka–Volterra 模拟数据：
data(block_3sp)  
# 数据框 block_3sp 含 198 行，10 列：time, x_t, x_t-1, x_t-2, y_t, … 等
head(block_3sp)

# 3. 单纯复形投影 (Simplex Projection) ---------------------------------------
smplx <- Simplex(
  dataFrame = block_3sp,
  lib       = "1 100",          # 用第 1–100 行作为训练库
  pred      = "101 190",        # 用第 101–190 行做预测
  E         = 3,                # 嵌入维度
  columns   = "x_t",            # 用 x_t 时间序列
  target    = "x_t",            # 预测目标也是 x_t
  showPlot  = FALSE
)
# 输出：一个 data.frame，含 Observations, Predictions 两列
head(smplx)
# 评估预测误差
err_simplex <- ComputeError(smplx$Observations, smplx$Predictions)
print(err_simplex)
# 返回列表：$rho (皮尔逊相关), $MAE, $RMSE :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}

# 4. S‑map（局部加权线性投影） ---------------------------------------------
smap <- SMap(
  dataFrame = block_3sp,
  lib       = "1 100",
  pred      = "101 190",
  E         = 3,
  theta     = 2,                # 非线性权重参数
  columns   = "x_t",
  target    = "x_t",
  showPlot  = FALSE,
  parameterList = TRUE          # 同时返回回归系数
)
# 输出：列表，含 $predictions (Observations & Predictions) 和 $coefficients (每个时刻的局部回归系数)
head(smap$predictions)
head(smap$coefficients)

# 5. 收敛交叉映射 (Convergent Cross Mapping, CCM) ----------------------------
# 测试 y_t → x_t 的映射
ccm_xy <- CCM(
  dataFrame   = block_3sp,
  columns     = "y_t",          # 用 y_t 构建库
  target      = "x_t",          # 用于预测 x_t
  E           = 3,
  libSizes    = "10 80 10",     # 库大小从 10 到 80，每次加 10
  sample      = 50,             # 每个库大小随机抽 50 次
  random      = TRUE,
  showPlot    = FALSE
)
# 输出：data.frame，含 LibSize, rho_xy (y→x), rho_yx (x→y)
print(ccm_xy)

# 6. 最优嵌入维度 (EmbedDimension) --------------------------------------------
edim <- EmbedDimension(
  dataFrame  = block_3sp,
  lib        = "1 150",
  pred       = "151 198",
  maxE       = 6,
  columns    = "x_t",
  target     = "x_t",
  showPlot   = FALSE
)
# 输出：data.frame，含 E, rho；可用于选择最佳 E
print(edim)

# 7. 多视图嵌入预测 (Multiview Embedding) -------------------------------------
mve <- Multiview(
  dataFrame   = block_3sp,
  lib         = "1 100",
  pred        = "101 190",
  E           = 2,
  D           = 3,              # 从 3 种变量组合中选
  columns     = "x_t y_t z_t",
  target      = "x_t",
  multiview   = 50,             # 平均前 50 个最佳组合
  showPlot    = FALSE
)
# 输出：列表 [[View, Predictions]]，其中 Predictions 包含预测值
head(mve$Predictions)

# 8. 非线性测试 (PredictNonlinear) --------------------------------------------
pn <- PredictNonlinear(
  dataFrame  = block_3sp,
  lib        = "1 100",
  pred       = "101 190",
  E          = 3,
  theta      = "0 1 2 4 8",     # 扫描不同 theta 值
  columns    = "x_t",
  target     = "x_t",
  showPlot   = FALSE
)
# 输出：data.frame，含 Theta, rho；可用于判断系统非线性强度
print(pn)

