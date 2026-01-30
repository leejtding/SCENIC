#!/usr/bin/env Rscript

suppressWarnings(suppressMessages({
  library(optparse)
  library(jsonlite)
  library(survival)
}))

option_list <- list(
  make_option(c("--method"), type="character", help="Method: icenreg | dnn_ic | icrf"),
  make_option(c("--sim_model"), type="character", help="PH | PO | AFT"),
  make_option(c("--p_cov"), type="integer", help="Number of covariates"),
  make_option(c("--train_csv"), type="character", help="Path to training CSV"),
  make_option(c("--tgrid_json"), type="character", help="Path to JSON list of time grid values"),
  make_option(c("--xtest_json"), type="character", help="Path to JSON 2D array of 4 x p_cov test covariates"),
  make_option(c("--out_json"), type="character", help="Path to write output JSON"),

  # ---- icenReg options (ignored unless method == icenreg) ----
  make_option(c("--icenreg_model"), type="character", default="PH",
              help="Model type for icenReg: PH | PO | AFT"),

  # ---- DNN-IC options (ignored unless method == dnn_ic) ----
  make_option(c("--dnn_epoch"), type="integer", default=1000),
  make_option(c("--dnn_batch_size"), type="integer", default=50),
  make_option(c("--dnn_num_nodes"), type="integer", default=50),
  make_option(c("--dnn_activation"), type="character", default="selu"),
  make_option(c("--dnn_l1"), type="double", default=0.1),
  make_option(c("--dnn_dropout"), type="double", default=0.0),
  make_option(c("--dnn_lr"), type="double", default=0.0002),
  make_option(c("--dnn_num_layer"), type="integer", default=2),
  make_option(c("--dnn_m"), type="integer", default=3),
  make_option(c("--dnn_fun"), type="character", default="fun_DNN-IC.R",
              help="Path to fun_DNN-IC.R (copied into temp dir by Python runner)"),

  # ---- ICRF options (ignored unless method == icrf) ----
  make_option(c("--icrf_ntree"), type="integer", default=300),
  make_option(c("--icrf_nodesize"), type="integer", default=6),
  make_option(c("--icrf_mtry"), type="integer", default=0,
              help="0 means use ceiling(sqrt(p_cov))"),
  make_option(c("--icrf_nfold"), type="integer", default=10),
  make_option(c("--icrf_split_rule"), type="character", default="Wilcoxon"),
  make_option(c("--icrf_quasihonesty"), type="logical", default=TRUE),
  make_option(c("--icrf_ert"), type="logical", default=TRUE),
  make_option(c("--icrf_uniform_ert"), type="logical", default=TRUE),
  make_option(c("--icrf_replace"), type="logical", default=FALSE),
  make_option(c("--icrf_sampsize_frac"), type="double", default=0.95),
  make_option(c("--icrf_bandwidth"), type="double", default=NA_real_,
              help="NA means use icrf default (smoothed); 0 forces no smoothing in the NPMLE smoother")
)

opt <- parse_args(OptionParser(option_list=option_list))

method <- opt$method
sim_model <- opt$sim_model
p_cov <- opt$p_cov

train <- read.csv(opt$train_csv)
tgrid <- fromJSON(opt$tgrid_json)
xtest <- fromJSON(opt$xtest_json)

# Covariate names expected from Python sim code: x1..xp
xnames <- paste0("x", 1:p_cov)

out <- list()

if (method == "icenreg") {

  # Parametric interval-censored regression via icenReg::ic_par
  if (!requireNamespace("icenReg", quietly=TRUE)) {
    stop("Package 'icenReg' not installed. Install.packages('icenReg').")
  }
  suppressWarnings(suppressMessages(library(icenReg)))

  icenreg_model <- opt$icenreg_model
  model_type <- switch(icenreg_model,
                       "PH"="ph",
                       "PO"="po",
                       "AFT"="aft",
                       stop("sim_model must be PH/PO/AFT"))

  rhs <- paste(xnames, collapse=" + ")
  fml <- as.formula(paste0("Surv(l, r, type='interval2') ~ ", rhs))
  newdata <- as.data.frame(xtest)
  colnames(newdata) <- xnames

  if (model_type == "ph" || model_type == "po") {
    fit <- icenReg::ic_sp(fml, data=train, model=model_type)
    sc <- icenReg::getSCurves(fit, newdata = newdata)
    ints <- sc$Tbull_ints

    s_on_grid <- lapply(sc$S_curves, function(sk) {
      idx <- findInterval(tgrid, ints[, "lower"], rightmost.closed = TRUE)
      idx[idx < 1] <- 1
      sk[idx]
    })
    s_hat <- do.call(rbind, s_on_grid)
  } else if (model_type == "aft") {
    fit <- icenReg::ic_par(fml, data=train, model=model_type, dist="weibull")
    s_hat <- sapply(tgrid, function(t) {
      getFitEsts(fit, newdata = newdata, q = t)
    })
  } else {
    stop("Unknown model_type for icenReg.")
  }

  s_hat <- as.matrix(s_hat)
  if (nrow(s_hat) != nrow(newdata) && ncol(s_hat) == nrow(newdata)) {
    s_hat <- t(s_hat)
  }

  out$method <- paste0("icenreg_", model_type)
  out$s_hat <- s_hat
  out$extra <- list(coefficients=fit$coefficients)

} else if (method == "dnn_ic") {

  # DNN-IC model from fun_DNN-IC.R (Keras/TensorFlow in R)
  suppressWarnings(suppressMessages({
    library(tensorflow)
    library(keras3)
    library(R6)
  }))

  # Source the functions (copied into temp dir by Python runner)
  source(opt$dnn_fun)

  # Training data produced by Python runner:
  # columns: Left, Right, status, x1..xp
  if (!all(c("Left", "Right", "status") %in% colnames(train))) {
    stop("For method dnn_ic, train_csv must contain columns: Left, Right, status, x1..xp")
  }

  left_dat   <- as.matrix(train[, "Left", drop=FALSE], nrow=nrow(train))
  right_dat  <- as.matrix(train[, "Right", drop=FALSE], nrow=nrow(train))
  status_dat <- as.matrix(train[, "status", drop=FALSE], nrow=nrow(train))
  pred_dat   <- as.matrix(train[, xnames, drop=FALSE], nrow=nrow(train))

  num_l <- min(left_dat)
  num_u <- max(right_dat)

  model <- build_model_ic(
    left_dat, right_dat, pred_dat,
    m = as.integer(opt$dnn_m), l = num_l, u = num_u,
    num_nodes = as.integer(opt$dnn_num_nodes),
    string_activation = opt$dnn_activation,
    num_l1 = as.numeric(opt$dnn_l1),
    num_dropout = as.numeric(opt$dnn_dropout),
    num_lr = as.numeric(opt$dnn_lr),
    num_layer = as.integer(opt$dnn_num_layer)
  )

  model %>% fit(
    list(left_dat, right_dat, pred_dat),
    status_dat,
    epochs = as.integer(opt$dnn_epoch),
    batch_size = as.integer(opt$dnn_batch_size),
    verbose = 0
  )

  # Predict survival on tgrid for each of 4 xtest rows
  newdata <- as.data.frame(xtest)
  colnames(newdata) <- xnames
  x_new <- as.matrix(newdata)

  # Get subject-specific linear predictor term from column 3 of model output
  # Provide dummy times (pred doesn't depend on time in that component)
  dummy_left  <- matrix(0, nrow=nrow(x_new), ncol=1)
  dummy_right <- matrix(1, nrow=nrow(x_new), ncol=1)
  out_pred <- model %>% predict(list(dummy_left, dummy_right, x_new))
  eta <- as.vector(out_pred[,3])
  a <- exp(eta)

  # For each subject, compute gL(t) by predicting at time=t and same x
  # Then S(t)=exp( - exp(pred_i) * gL(t) )
  s_hat <- matrix(NA_real_, nrow=nrow(x_new), ncol=length(tgrid))
  tmat <- matrix(tgrid, ncol=1)

  for (i in 1:nrow(x_new)) {
    Xi <- matrix(x_new[i, ], nrow=length(tgrid), ncol=ncol(x_new), byrow=TRUE)
    out_time <- model %>% predict(list(tmat, tmat, Xi))
    gL <- as.vector(out_time[,1])
    s_hat[i, ] <- exp(- a[i] * gL)
  }

  out$method <- "dnn_ic"
  out$s_hat <- s_hat
  out$extra <- list(
    epoch=as.integer(opt$dnn_epoch),
    batch_size=as.integer(opt$dnn_batch_size),
    num_nodes=as.integer(opt$dnn_num_nodes),
    activation=opt$dnn_activation,
    l1=as.numeric(opt$dnn_l1),
    dropout=as.numeric(opt$dnn_dropout),
    lr=as.numeric(opt$dnn_lr),
    num_layer=as.integer(opt$dnn_num_layer),
    m=as.integer(opt$dnn_m)
  )

} else if (method == "icrf") {

  # Interval-Censored Random Forest (ICRF) from the 'icrf' package
  if (!requireNamespace("icrf", quietly=TRUE)) {
    stop("Package 'icrf' not installed. Install.packages('icrf').")
  }
  suppressWarnings(suppressMessages(library(icrf)))

  # Accept either (l,r) or (L,R) columns from Python
  if (all(c("l","r") %in% colnames(train))) {
    Lcol <- "l"; Rcol <- "r"
  } else if (all(c("L","R") %in% colnames(train))) {
    Lcol <- "L"; Rcol <- "R"
  } else if (all(c("Left","Right") %in% colnames(train))) {
    Lcol <- "Left"; Rcol <- "Right"
  } else {
    stop("For method icrf, train_csv must contain interval columns named (l,r) or (L,R) or (Left,Right).")
  }

  # Build time grid for smoothing inside icrf
  tgrid_num <- as.numeric(tgrid)
  tau <- max(tgrid_num[is.finite(tgrid_num)])
  time_smooth <- sort(unique(c(tgrid_num[is.finite(tgrid_num)], Inf)))

  x_train <- as.data.frame(train[, xnames, drop=FALSE])
  L <- as.numeric(train[[Lcol]])
  R <- as.numeric(train[[Rcol]])

  mtry_use <- as.integer(opt$icrf_mtry)
  if (is.na(mtry_use) || mtry_use <= 0) mtry_use <- ceiling(sqrt(p_cov))

  sampsize <- floor(nrow(train) * as.numeric(opt$icrf_sampsize_frac))

  bw_opt <- opt$icrf_bandwidth
  bandwidth <- NULL
  if (!is.null(bw_opt)) {
    # accept "NA", "NULL", "" as "no bandwidth supplied"
    if (is.character(bw_opt) && tolower(trimws(bw_opt)) %in% c("na", "null", "")) {
      bandwidth <- NULL
    } else {
      bandwidth <- as.numeric(bw_opt)
      if (length(bandwidth) != 1 || is.na(bandwidth) || !is.finite(bandwidth) || bandwidth <= 0) {
        stop("icrf_bandwidth must be NULL/NA/'' or a single finite positive number.")
      }
    }
  }

  set.seed(1)
  fit <- icrf:::icrf.default(
    x = x_train,
    L = L, R = R,
    tau = tau,
    timeSmooth = time_smooth,
    proximity = FALSE,
    importance = FALSE,
    nPerm = 10,
    nfold = as.integer(opt$icrf_nfold),
    ntree = as.integer(opt$icrf_ntree),
    nodesize = as.integer(opt$icrf_nodesize),
    mtry = mtry_use,
    replace = as.logical(opt$icrf_replace),
    sampsize = sampsize,
    split.rule = opt$icrf_split_rule,
    ERT = as.logical(opt$icrf_ert),
    uniformERT = as.logical(opt$icrf_uniform_ert),
    quasihonesty = as.logical(opt$icrf_quasihonesty),
    initialSmoothing = FALSE,
    bandwidth = bandwidth
  )

  newdata <- as.data.frame(xtest)
  colnames(newdata) <- xnames

  pr <- predict(fit, newdata = newdata, smooth = TRUE)
  pr <- as.matrix(pr)

  # Interpolate predictions to the requested tgrid (exclude Inf)
  tp <- as.numeric(fit$time.points.smooth)
  finite_idx <- is.finite(tp)
  tp_f <- tp[finite_idx]

  s_hat <- matrix(NA_real_, nrow=nrow(newdata), ncol=length(tgrid_num))
  for (i in 1:nrow(newdata)) {
    s_i <- as.numeric(pr[i, finite_idx])
    s_i <- pmin(1, pmax(0, s_i))
    s_hat[i, ] <- approx(x=tp_f, y=s_i, xout=tgrid_num, method="linear", rule=2, ties="ordered")$y
  }

  out$method <- "icrf"
  out$s_hat <- s_hat
  out$extra <- list(
    ntree=as.integer(opt$icrf_ntree),
    nodesize=as.integer(opt$icrf_nodesize),
    mtry=mtry_use,
    nfold=as.integer(opt$icrf_nfold),
    split_rule=opt$icrf_split_rule,
    quasihonesty=as.logical(opt$icrf_quasihonesty),
    ERT=as.logical(opt$icrf_ert),
    sampsize=sampsize
  )

} else {
  stop(paste0("Unknown method: ", method))
}

write(toJSON(out, auto_unbox=TRUE), file=opt$out_json)
