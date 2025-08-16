# STAT580 Project 2

# training dataset
dt1=read.csv("CollegeCr.csv", header=T)
dt2=read.csv("Edwards.csv", header=T)
dt3=read.csv("OldTown.csv", header=T)

# testing dataset
test1=read.csv("CollegeCr.test.csv", header=T)
test2=read.csv("Edwards.test.csv", header=T)
test3=read.csv("OldTown.test.csv", header=T)

# ---------------------------------------------
# 0. data cleaning
# ---------------------------------------------
# add neighborhood information
dt1$Neighbor="CollegeCr"
dt2$Neighbor="Edwards"
dt3$Neighbor="OldTown"

test1$Neighbor="CollegeCr"
test2$Neighbor="Edwards"
test3$Neighbor="OldTown"

# names(dt1)
# names(dt2)
# names(dt3)

# (0)combine three datasets: dt_all, test_all
library(dplyr)
dt_all <- bind_rows(dt1, dt2, dt3)
write.csv(dt_all, file = "train_all.csv", row.names = FALSE)

test_all <- bind_rows(test1, test2, test3)
write.csv(test_all, file = "test_all.csv", row.names = FALSE)

# (1)delete nan columns: dt_new, test_new
del=c("Street", "LandContour", "grade to building","LandSlope",
     "YearRemodAdd","BsmtExposure","BsmtUnfSF","KitchenAbvGr")
dt_new <- dt_all %>%
  select(-any_of(del))
test_new <- test_all %>%
  select(-any_of(del))

# (2)split concatenated variables: dt_new1, test_new1
library(tidyr)
dt_new1 <- dt_new %>%
  separate(Exterior, into = c("Exterior1st", "ExterQual", "ExterCond"), sep = ";") %>%
  separate(LotInfo, into = c("LotConfig", "LotShape", "LotFrontage", "LotArea"), sep = ";")
test_new1 <- test_new %>%
  separate(Exterior, into = c("Exterior1st", "ExterQual", "ExterCond"), sep = ";") %>%
  separate(LotInfo, into = c("LotConfig", "LotShape", "LotFrontage", "LotArea"), sep = ";")

# (3)nan or empty values: dt_new1, test_new1
dt_new1 <- dt_new1 %>%
  mutate(
    # 1. GarageType：空值 → "noGarage"
    GarageType = ifelse(is.na(GarageType) | GarageType == "", "noGarage", GarageType),
    
    # 2. LotFrontage：转为数值，NA → 均值
    LotFrontage = as.numeric(LotFrontage),
    LotFrontage = ifelse(is.na(LotFrontage),
                         mean(LotFrontage, na.rm = TRUE),
                         LotFrontage),
    
    # 3. LotArea：转为数值，NA → 均值
    LotArea = as.numeric(LotArea),
    LotArea = ifelse(is.na(LotArea),
                     mean(LotArea, na.rm = TRUE),
                     LotArea),
    
    # 4. BsmtCond：NA → "noBasement"
    BsmtCond = ifelse(is.na(BsmtCond) | BsmtCond == "", "noBasement", BsmtCond)
  )

test_new1 <- test_new1 %>%
  mutate(
    # 1. GarageType：空值 → "noGarage"
    GarageType = ifelse(is.na(GarageType) | GarageType == "", "noGarage", GarageType),
    
    # 2. LotFrontage：转为数值，NA → 均值
    LotFrontage = as.numeric(LotFrontage),
    LotFrontage = ifelse(is.na(LotFrontage),
                         mean(LotFrontage, na.rm = TRUE),
                         LotFrontage),
    
    # 3. LotArea：转为数值，NA → 均值
    LotArea = as.numeric(LotArea),
    LotArea = ifelse(is.na(LotArea),
                     mean(LotArea, na.rm = TRUE),
                     LotArea),
    
    # 4. BsmtCond：NA → "noBasement"
    BsmtCond = ifelse(is.na(BsmtCond) | BsmtCond == "", "noBasement", BsmtCond)
  )



# (4) houseAge=YrSold-YearBuilt, then delete YrSold and YearBuilt: dt_new1, test_new1
# age<0, 判断为记录反了卖出时间和建造时间，因此取绝对值，保留该样本数据
dt_new1 <- dt_new1 %>%
  mutate(houseAge = YrSold - YearBuilt,
         houseAge = ifelse(houseAge < 0, abs(houseAge), houseAge)) %>%
  select(-YrSold, -YearBuilt)                 # 删除旧列
summary(dt_new1)

test_new1 <- test_new1 %>%
  mutate(houseAge = YrSold - YearBuilt,
         houseAge = ifelse(houseAge < 0, abs(houseAge), houseAge)) %>%
  select(-YrSold, -YearBuilt)                 # 删除旧列
summary(test_new1)

# (5) check 哪些列仍存在 脏数据
library(stringr)
# 定义函数判断一列中“脏数据”的数量
count_dirty <- function(col) {
  if (is.factor(col)) col <- as.character(col)  # 因子转字符，方便判断
  sum(
    is.na(col) |                       # NA
      col == "NA" |                     # 字符串"NA"
      col == "" |                       # 空字符串
      str_trim(col) == "" |             # 全空格字符串
      (is.numeric(col) & is.nan(col))   # NaN（仅数值列）
    , na.rm = TRUE)
}

# 计算每列脏数据个数: dt_new1
dirty_counts <- sapply(dt_new1, count_dirty)

# 筛选有脏数据的列和对应数量: dt_new1
dirty_cols <- dirty_counts[dirty_counts > 0]
dirty_cols

# test dataset: test_new1
test_dirty_counts <- sapply(test_new1, count_dirty)
test_dirty_cols <- test_dirty_counts[test_dirty_counts > 0]
test_dirty_cols


# function函数：提取众数
get_mode <- function(v) {
  v <- v[!is.na(v)]
  if(length(v) == 0) return(NA)
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

# 对仍存在脏数据的列进行填充
for (col_name in names(dirty_cols)) {
  if (is.numeric(dt_new1[[col_name]])) {
    # 数值列均值填充
    mean_val <- mean(dt_new1[[col_name]], na.rm = TRUE)
    dt_new1[[col_name]][
      is.na(dt_new1[[col_name]]) | is.nan(dt_new1[[col_name]])] <- mean_val
  } else {
    # 分类列众数填充
    mode_val <- get_mode(dt_new1[[col_name]])
    # 先把字符型的 "NA", "", " " 替换成 NA
    dt_new1[[col_name]][dt_new1[[col_name]] %in% c("NA", "", " ")] <- NA
    dt_new1[[col_name]][is.na(dt_new1[[col_name]])] <- mode_val
  }
}

# 对仍存在脏数据的列进行填充: test_new1
for (col_name in names(test_dirty_cols)) {
  if (is.numeric(test_new1[[col_name]])) {
    # 数值列均值填充
    mean_val <- mean(test_new1[[col_name]], na.rm = TRUE)
    test_new1[[col_name]][
      is.na(test_new1[[col_name]]) | is.nan(test_new1[[col_name]])] <- mean_val
  } else {
    # 分类列众数填充
    mode_val <- get_mode(test_new1[[col_name]])
    # 先把字符型的 "NA", "", " " 替换成 NA
    test_new1[[col_name]][test_new1[[col_name]] %in% c("NA", "", " ")] <- NA
    test_new1[[col_name]][is.na(test_new1[[col_name]])] <- mode_val
  }
}

# 合并train+test, then one-hot encoding
train_test_all <- bind_rows(dt_new1, test_new1)

library(fastDummies)
cat_vars <- names(train_test_all)[sapply(train_test_all, function(x) is.factor(x) | is.character(x))]
cat_vars
cat_vars <- setdiff(cat_vars, "uniqueID")
cat_vars

train_test_encoded <- fastDummies::dummy_cols(
  train_test_all,
  select_columns = cat_vars,
  remove_selected_columns = TRUE,
  remove_first_dummy = TRUE   # 只保留k-1个哑变量，适合线性回归
)

# split into train + test
dt_new1_encoded  <- train_test_encoded %>%
  filter(is.na(uniqueID) | uniqueID == 0 | uniqueID == "NA") %>%
  select(-uniqueID)
test_new1_encoded <- train_test_encoded %>% filter(uniqueID != 0) %>%
  select(-SalePrice)
# dt_new1_encoded 为编码后的train集，用于线性回归
# test_new1_encoded: 编码后的test集 


# (6) 对train样本数据标号: dt_new1_encoded
dt_new1_encoded <- dt_new1_encoded %>%
  mutate(sample_id = row_number())

write.csv(dt_new1_encoded, file = "dt_new1_encoded.csv", row.names = FALSE)
# dt_new1_encoded.csv 为清洗编码好的train数据集
write.csv(test_new1_encoded, file = "test_new1_encoded.csv", row.names = FALSE)
# test_new1_encoded.csv 为清洗编码好的test数据集



# ----------------------
# 1.EDA
# ----------------------
# (1) numerical variables
num_vars <- names(dt_new1)[sapply(dt_new1, is.numeric)]
num_vars
# group by neighborhood
num_summary <- dt_new1 %>%
  group_by(Neighbor) %>%
  summarise_at(num_vars, mean, na.rm = TRUE)
print(num_summary)


# more summary information
summary_stats <- dt_new1 %>%
  select(all_of(num_vars)) %>%
  summarise(across(everything(), list(
    count = ~sum(!is.na(.)),
    mean = ~mean(., na.rm=TRUE),
    median = ~median(., na.rm=TRUE),
    sd = ~sd(., na.rm=TRUE),
    min = ~min(., na.rm=TRUE),
    max = ~max(., na.rm=TRUE),
    q1 = ~quantile(., 0.25, na.rm=TRUE),
    q3 = ~quantile(., 0.75, na.rm=TRUE)
  ))) %>%
  tidyr::pivot_longer(everything(),
                      names_to = c("variable", "stat"),
                      names_pattern = "(.*)_(count|mean|median|sd|min|max|q1|q3)$",
                      values_to = "value")

#print(summary_stats)

library(dplyr)
library(tidyr)
library(knitr)

# 假设 summary_stats 是之前的长格式数据框
summary_wide <- summary_stats %>%
  pivot_wider(names_from = stat, values_from = value)

# 打印成表格（控制台或Rmarkdown友好）
print(kable(summary_wide, digits = 3, caption = "Numerical Variables Summary Statistics"))


# (2) category variables -------------------------
cat_vars <- names(dt_new1)[sapply(dt_new1, function(x) is.factor(x) | is.character(x))]
cat_vars

# 统计每个分类变量的频数和缺失情况
cat_summary <- lapply(cat_vars, function(var) {
  dt_new1 %>%
    count(!!sym(var), name = "count") %>%
    mutate(variable = var) %>%
    arrange(desc(count))
})

names(cat_summary) <- cat_vars
print(cat_summary)

# library(ggplot2)
# for (var in cat_vars) {
#   p <- ggplot(dt_new1, aes_string(x = var)) +
#     geom_bar(fill = "skyblue") +
#     theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
#     ggtitle(paste("Barplot of", var))
#   print(p)
# }


# -------------------------------
# 2.encode categorical variables
# -------------------------------
# library(fastDummies)
# cat_vars <- names(dt_new1)[sapply(dt_new1, function(x) is.factor(x) | is.character(x))]
# dt_new1_encoded <- fastDummies::dummy_cols(
#   dt_new1,
#   select_columns = cat_vars,
#   remove_selected_columns = TRUE,
#   remove_first_dummy = TRUE   # 只保留k-1个哑变量，适合线性回归
# )
# dt_new1_encoded 为编码后的数据集，用于线性回归


# ----------------------
# 3. linear regression
# ----------------------
# MLR with all variables===========================
hp_m1 <- lm(SalePrice ~ . - sample_id, data = dt_new1_encoded)
summary(hp_m1)

# assumption check==================
library(lmtest) 
library(Hmisc)
library(nortest)
library(car)
# 1 Normality check(not satisfied) 
# with QQ plot and Anderson-Darling Test
qqPlot(hp_m1$residuals, main="QQ Plot")# Deviations from qqline: departures from normality
ad.test(hp_m1$residuals)
shapiro.test(hp_m1$residuals)# p-value<0.05: not normally distributed

# 2 Constant variance check(not satisfied)
# with Res vs Fits plot and Breusch Pagan test
plot(hp_m1$fitted,hp_m1$residuals, ylab="Residuals", xlab="Fitted Values")
abline(0,0)# residuals increase when fitted value increases.
bptest(hp_m1)#Breusch-Pagan for constant variance# p-value<0.05: non-constant variance

# 3 Linearity check
rcorr(as.matrix(dt_new1_encoded))
# some p-value<0.05, many variables are correlated
# some p-value (price~variables) >0.05, so these doesn't have a liner relationship with price: 
#     OverallCond, HouseStyle_1Story, HouseStyle_2.5Story,SaleType_WD, RoofStyle_notGable,ExterQual_TA,
#     Heating_GasW,Heating_OthW,Heating_Wall,LotConfig_CulDSac,LotConfig_FR2,LotConfig_Inside,LotShape_IR3,
#     BldgType_Duplex,BldgType_Twnhs
# other variables all have a significant linear correlation with price.

library(ggplot2)
vif(hp_m1)
# check for multicollinearity among predictor variables# all VIF < 10, no multicollinearity
# error: because Utilities_ 变量是完全线性依赖

alias(hp_m1)#查看共线性变量: Utilities_ 变量是完全线性依赖（被其他变量完美预测），导致模型的共线性
# 从数据中删除 Utilities_ 变量,重新建模
dt_new1_encoded <- dt_new1_encoded %>% select(-Utilities_)
model <- lm(SalePrice ~ . - sample_id, data = dt_new1_encoded)
alias(model)
vif_values<-vif(model)
# 取出大于10的变量
high_vif <- vif_values[vif_values > 10]
print(high_vif)

# 4 Independence check(satisfied)
plot(model$residuals, main="Residuals vs Order(ykz5645)")
acf(model$residuals)# no autocorrelation: no any significant line beyond lag 0
dwtest(model)# DW=2: no autocorrelation# p-value>0.05: no autocorrelation (independent residuals)


# modify: log(SalePrice)
dt_new2_encoded$log_SalePrice <- log(dt_new1_encoded$SalePrice)
model_log <- lm(log_SalePrice ~ . - sample_id - SalePrice, data = dt_new2_encoded)
summary(model_log)
# check assumption again: problem still serious, 放弃MLR, try正则化回归
shapiro.test(model_log$residuals)
bptest(model_log)
dwtest(model_log)
vif_values<-vif(model_log)
high_vif <- vif_values[vif_values > 10]
print(high_vif)



# --------------------------------------------------
# 4.正则化回归
# ---------------------------------------------------
library(glmnet)
library(caret)
set.seed(12)

# ==============================
# method 2: cross validation without feature selection 
# ===============================
# 从数据中删除 Utilities_ 变量,重新建模: dt_new1_encoded
dt_new1_encoded <- dt_new1_encoded %>% select(-Utilities_)

# 去除非预测变量
X <- dt_new1_encoded %>% select(-SalePrice, -sample_id) %>% as.matrix()
y <- dt_new1_encoded$SalePrice

# response做log变换，缓解异方差
# y_log <- log(y)# e 
# mean(y_log)
# mean(y)
# y <- y_log

# ------------------------------
# k-fold 交叉验证
# caret训练控制：5折交叉验证，重复1次
train_control <- trainControl(method = "cv", number = 10)

# 1. Ridge回归 (alpha=0)
ridge_model <- train(
  x = X, y = y,
  method = "glmnet",
  trControl = train_control,
  tuneGrid = expand.grid(alpha = 0, lambda = 10^seq(-4, 1, length = 100))
)

# 2. Lasso回归 (alpha=1)
lasso_model <- train(
  x = X, y = y,
  method = "glmnet",
  trControl = train_control,
  tuneGrid = expand.grid(alpha = 1, lambda = 10^seq(-4, 1, length = 100))
)

# 3. Elastic Net (alpha在0~1之间自动调优)
enet_model <- train(
  x = X, y = y,
  method = "glmnet",
  trControl = train_control,
  tuneLength = 10   # 自动搜索alpha和lambda
)

# 查看结果
print(ridge_model)
print(lasso_model)
print(enet_model)

# 比较最优模型的RMSE（caret默认度量）
cat("Best Ridge RMSE:", min(ridge_model$results$RMSE), "\n")
cat("Best Lasso RMSE:", min(lasso_model$results$RMSE), "\n")
cat("Best Elastic Net RMSE:", min(enet_model$results$RMSE), "\n")

# Result 1(can not rebuild)-----------------------
# Best Ridge RMSE: 26826.93
# Best Lasso RMSE: 17659.41 
# Best Elastic Net RMSE: 17003.36 (BEST! one)

# Result 2-----------------------
# Best Ridge RMSE: 44364.41  
# Best Lasso RMSE: 52453.66   
# Best Elastic Net RMSE: 37954.48   (BEST!)


# ==============================
# method 3: feature selection (delete high VIF + standardize)
# ===============================
library(caret)
library(car)
# 先去掉响应变量
dt_no_id <- dt_new1_encoded[, !(names(dt_new1_encoded) %in% "sample_id")]

# 计算 VIF
lm_for_vif <- lm(SalePrice ~ ., data = dt_no_id)
vif_values <- vif(lm_for_vif)
high_vif_vars <- names(vif_values[vif_values > 10])
cat("VIF>10 的变量:\n", high_vif_vars, "\n")

# 删除 VIF>10 的变量
dt_reduced <- dt_no_id[, !(names(dt_no_id) %in% high_vif_vars)]


# 2. 标准化特征
# 分离响应变量
y <- dt_reduced$SalePrice
x <- dt_reduced[, !(names(dt_reduced) %in% "SalePrice")]
# 标准化特征
preProc <- preProcess(x, method = c("center", "scale"))
x_scaled <- predict(preProc, x)

# 合并回去
dt_scaled <- cbind(SalePrice = y, x_scaled)

# 3. k-fold 交叉验证设置
set.seed(12)
ctrl <- trainControl(method = "cv", number = 10)

# 4. 岭回归
ridge_model <- train(
  SalePrice ~ ., data = dt_scaled,
  method = "ridge",
  tuneLength = 20,
  trControl = ctrl,
  metric = "RMSE"
)

# 5. Lasso
lasso_model <- train(
  SalePrice ~ ., data = dt_scaled,
  method = "lasso",
  tuneLength = 20,
  trControl = ctrl,
  metric = "RMSE"
)

# 6. Elastic Net
enet_model <- train(
  SalePrice ~ ., data = dt_scaled,
  method = "enet",
  tuneLength = 20,
  trControl = ctrl,
  metric = "RMSE"
)

# 7. 输出 RMSE 结果
cat("Best Ridge RMSE:", min(ridge_model$results$RMSE), "\n")
cat("Best Lasso RMSE:", min(lasso_model$results$RMSE), "\n")
cat("Best Elastic Net RMSE:", min(enet_model$results$RMSE), "\n")

# result
print(enet_model)

# result 1--------------------
# Best Ridge RMSE: 92950.37 
# Best Lasso RMSE: 36128.4 
# Best Elastic Net RMSE: 35107.84 

# result 2--------------------
# Best Ridge RMSE: 55953.55  
# Best Lasso RMSE: 35596.42  
# Best Elastic Net RMSE: 34904.15 (best) 

# 
# # ==============================
# # method 3: feature selection (standardize only)
# # ===============================
# library(caret)
# library(car)
# 
# # 1. 标准化特征
# # 分离响应变量
# y <- dt_new1_encoded$SalePrice
# x <- dt_new1_encoded[, !(names(dt_new1_encoded) %in% "SalePrice")]
# # 标准化特征
# preProc <- preProcess(x, method = c("center", "scale"))
# x_scaled <- predict(preProc, x)
# 
# # 合并回去
# dt_scaled <- cbind(SalePrice = y, x_scaled)
# 
# # 2. k-fold 交叉验证设置
# set.seed(12)
# ctrl <- trainControl(method = "cv", number = 10)
# 
# # 4. 岭回归
# ridge_model <- train(
#   SalePrice ~ ., data = dt_scaled,
#   method = "ridge",
#   tuneLength = 20,
#   trControl = ctrl,
#   metric = "RMSE"
# )
# 
# # 5. Lasso
# lasso_model <- train(
#   SalePrice ~ ., data = dt_scaled,
#   method = "lasso",
#   tuneLength = 20,
#   trControl = ctrl,
#   metric = "RMSE"
# )
# 
# # 6. Elastic Net
# enet_model <- train(
#   SalePrice ~ ., data = dt_scaled,
#   method = "enet",
#   tuneLength = 20,
#   trControl = ctrl,
#   metric = "RMSE"
# )
# 
# # 7. 输出 RMSE 结果
# cat("Best Ridge RMSE:", min(ridge_model$results$RMSE), "\n")
# cat("Best Lasso RMSE:", min(lasso_model$results$RMSE), "\n")
# cat("Best Elastic Net RMSE:", min(enet_model$results$RMSE), "\n")
# 
# # result 1--------------------
# # Best Ridge RMSE: 64470.06  
# # Best Lasso RMSE: 35111.69  
# # Best Elastic Net RMSE: 34482.57 


# ==================================================
# test
# ==================================================
# 使用最优模型预测: enet_model 没有feature selection
test_preds <- predict(enet_model, newdata = test_new1_encoded)

# 导出预测结果
output <- data.frame(uniqueID = test_new1_encoded$uniqueID, SalePrice = exp(1)^test_preds)
write.csv(output, "enet_predictions2.csv", row.names = FALSE)


# #---
# # test_new1_encoded 中有，但 x 中没有的列
# cols_in_test_not_in_train <- setdiff(names(test_new1_encoded), names(dt_new1_encoded))
# 
# # x 中有，但 test_new1_encoded 中没有的列
# cols_in_train_not_in_test <- setdiff(names(dt_new1_encoded), names(test_new1_encoded))
# 
# # 打印结果
# cat("测试集多出的列:\n")
# print(cols_in_test_not_in_train)
# 
# cat("训练集多出的列:\n")
# print(cols_in_train_not_in_test)


# ==============================
# method 1: one time training 
# ===============================

# # 划分训练测试集
# set.seed(123)
# train_idx <- sample(seq_len(nrow(X)), size = 0.8*nrow(X))
# 
# X_train <- X[train_idx, ]
# X_test  <- X[-train_idx, ]
# 
# y_train <- y_log[train_idx]
# y_test  <- y_log[-train_idx]
# 
# # (1) ridge regression 岭回归
# cv_ridge <- cv.glmnet(X_train, y_train, alpha = 0)
# 
# # 最佳lambda
# best_lambda_ridge <- cv_ridge$lambda.min
# 
# # 训练最终模型
# model_ridge <- glmnet(X_train, y_train, alpha = 0, lambda = best_lambda_ridge)
# 
# # 预测测试集
# pred_ridge <- predict(model_ridge, X_test)
# 
# 
# # (2) lasso(alpha=1)
# cv_lasso <- cv.glmnet(X_train, y_train, alpha = 1)
# 
# best_lambda_lasso <- cv_lasso$lambda.min
# model_lasso <- glmnet(X_train, y_train, alpha = 1, lambda = best_lambda_lasso)
# 
# pred_lasso <- predict(model_lasso, X_test)
# 
# 
# # (3) elastic
# set.seed(123)
# 
# # 用 caret 或自己写循环寻找最优alpha和lambda
# # 简单示范，用 alpha=0.5
# 
# cv_enet <- cv.glmnet(X_train, y_train, alpha = 0.5)
# 
# best_lambda_enet <- cv_enet$lambda.min
# model_enet <- glmnet(X_train, y_train, alpha = 0.5, lambda = best_lambda_enet)
# 
# pred_enet <- predict(model_enet, X_test)
# 
# 
# # (4) model evaluation
# rmse <- function(true, pred) (mean((true - pred)^2))^0.5
# 
# rmse_ridge <- rmse(y_test, pred_ridge)
# rmse_lasso <- rmse(y_test, pred_lasso)
# rmse_enet  <- rmse(y_test, pred_enet)
# 
# cat("Ridge MSE:", rmse_ridge, "\n")
# cat("Lasso MSE:", rmse_lasso, "\n")
# cat("Elastic Net MSE:", rmse_enet, "\n")




# ==============================
# method 4: xgboost
# ===============================

library(caret)
# library(xgboost)

# 1. 拆分特征和响应
X <- dt_new1_encoded %>% select(-SalePrice) %>% select(-sample_id)   # 特征
y <- dt_new1_encoded$SalePrice               # 响应变量

# # if删除 VIF>10 的变量
# X <- X[, !(names(X) %in% high_vif_vars)]

# 2. 设置交叉验证, 10-fold
train_control <- trainControl(
  method = "cv",
  number = 10,
  verboseIter = TRUE
)

# 3. 训练 Boosted Trees 模型 (xgboost)
xgb_grid <- expand.grid(
  nrounds = c(100, 200),           # boosting 迭代次数
  max_depth = c(3, 5, 7),          # 树的最大深度
  eta = c(0.05, 0.1, 0.3),         # 学习率
  gamma = 0,                       # 节点分裂最小损失
  colsample_bytree = 0.8,          # 每棵树的列采样比例
  min_child_weight = 1,            # 叶子节点的最小样本权重
  subsample = 0.8                  # 每棵树的样本采样比例
)

set.seed(12)
xgb_model <- train(
  x = X,
  y = y,
  method = "xgbTree",
  trControl = train_control,
  tuneGrid = xgb_grid,
  metric = "RMSE"
)

# 4. 输出最优模型和 RMSE
print(xgb_model)
cat("Best RMSE:", min(xgb_model$results$RMSE), "\n")
print(xgb_model$results)

# 5. 如果要在测试集上预测
test_new3_encoded <- test_new1_encoded %>% select(-uniqueID)   %>% select(-Utilities_)
pred <- predict(xgb_model, newdata = test_new3_encoded)

# 导出预测结果
output <- data.frame(uniqueID = test_new1_encoded$uniqueID, SalePrice = pred)
write.csv(output, "xgb_predictions2.csv", row.names = FALSE)




# ==============================
# model interpretation
# ===============================
# (1) feature importance
library(xgboost)
library(caret)

# (01) 获取变量重要性
importance <- varImp(xgb_model)# 最重要的变量标准化到 100%，其他变量的分数都是相对于它的百分比。
plot(importance, top = 20)  # 前 20 个重要变量


# 拿训练好的模型，比如你的 finalModel
importance_matrix <- xgb.importance(model = xgb_model$finalModel)
print(importance_matrix)
xgb.plot.importance(importance_matrix, top_n = 20, measure = "Gain")
# 选前20个变量
top_vars <- importance_matrix[1:20, ]

# 用ggplot画横向条形图
ggplot(top_vars, aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_bar(stat = "identity", fill = "#660099") +
  geom_text(aes(label = round(Gain, 3)), 
            hjust = -0.1,  # 数字稍微靠右
            size = 4) +
  coord_flip() +
  labs(
    title = "XGBoost Feature Importance (Gain)",
    x = "Feature",
    y = "Gain"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold"),
    axis.title.y = element_text(margin = margin(r = 10)),
    axis.title.x = element_text(margin = margin(t = 10))
  ) +
  expand_limits(y = max(top_vars$Gain) * 1.1)  # 给右边留点空隙放数字

library(SHAPforxgboost)
# 转换成 xgboost 原生模型
best_params <- xgb_model$bestTune
dtrain <- xgb.DMatrix(as.matrix(X), label = y)
xgb_native <- xgboost(
  data = dtrain,
  nrounds = best_params$nrounds,
  max_depth = best_params$max_depth,
  eta = best_params$eta,
  gamma = best_params$gamma,
  colsample_bytree = best_params$colsample_bytree,
  min_child_weight = best_params$min_child_weight,
  subsample = best_params$subsample,
  objective = "reg:squarederror"
)

# 计算 SHAP
shap_values <- shap.values(xgb_native, X_train = as.matrix(X))
shap_long <- shap.prep(shap_contrib = shap_values$shap_score, X_train = as.matrix(X))
shap.plot.summary(shap_long)

