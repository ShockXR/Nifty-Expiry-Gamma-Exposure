# Literature review: expiry-day volatility, dealer gamma and intraday forecasting

This review supports the Nifty expiry prototype. Every reference below was checked on 2 November 2026 against a publisher page, RePEc/IDEAS, SSRN, SEBI or NSE. The status column says how complete that check was. "Verified" means title, authors, year, venue and DOI (or a stable URL for working papers) were all confirmed. "Partly verified" names what is missing. Nothing here was taken on trust from memory.

## 1. The anchor paper

| Reference | Status | Why it matters here |
| --- | --- | --- |
| Chhimwal, B., Pandey, V. and Pandey, P. (2025). Effect of multiple index derivative expiry on volatility, volume, and connectedness: a tale of two stock indices in India. *Review of Derivatives Research*, 28(3), article 14. doi:10.1007/s11147-025-09221-8 | Verified | The paper this prototype extends. It shows expiry-day volume and volatility rose after Bank Nifty's expiry moved to Wednesday, and that Nifty and Bank Nifty volatility connectedness fell. It documents *that* the effect exists; it does not test *why* and makes no out-of-sample forecasts. See [PAPER_DEEP_DIVE.md](PAPER_DEEP_DIVE.md). |

## 2. Expiry-day effects

| Reference | Status | Relation to this project |
| --- | --- | --- |
| Stoll, H. R. and Whaley, R. E. (1987). Program trading and expiration-day effects. *Financial Analysts Journal*, 43(2), 16 to 28. doi:10.2469/faj.v43.n2.16 | Verified | The original "triple witching" evidence: abnormal volume, volatility and reversals at expiry. |
| Stoll, H. R. and Whaley, R. E. (1991). Expiration-day effects: What has changed? *Financial Analysts Journal*, 47(1), 58 to 72. doi:10.2469/faj.v47.n1.58 | Verified | Shows how rule changes altered expiry effects. The same before/after logic as the anchor paper, and as our regime tests (20 Nov 2024, 1 Sep 2025). |
| Vipul (2005). Futures and options expiration-day effects: The Indian evidence. *Journal of Futures Markets*, 25(11), 1045 to 1065. doi:10.1002/fut.20178 | Verified | Early NSE evidence on expiry-day price, volume and volatility effects. |
| Agarwalla, S. K. and Pandey, A. (2013). Expiration-day effects and the impact of short trading breaks on intraday volatility: Evidence from the Indian market. *Journal of Futures Markets*, 33(11), 1046 to 1070. doi:10.1002/fut.21632 | Verified from listings (volume, issue and pages added 3 November 2026 from Wiley and RePEc listings quoted by web search; see section 8) | Closest Indian precedent for an *intraday* expiry design: volatility rises in the last half hour of expiry days. Supports our focus on the last hour. |
| Singh, G. and Shaik, M. (2020). Re-examining the expiration effects of index futures: Evidence from India. *International Journal of Economics and Financial Issues*, 10(3), 16 to 23. https://ideas.repec.org/a/eco/journ1/2020-03-3.html | Partly verified (no DOI exists) | Before/after study around Bank Nifty weeklies. Lower-tier journal; cited only for context. |

## 3. Dealer hedging, gamma and pinning (the mechanism we test)

| Reference | Status | Relation to this project |
| --- | --- | --- |
| Avellaneda, M. and Lipkin, M. D. (2003). A market-induced mechanism for stock pinning. *Quantitative Finance*, 3(6), 417 to 425. doi:10.1088/1469-7688/3/6/301 | Verified | Theory: delta hedging by long-gamma market makers pulls the price toward a strike near expiry. Motivates the max-OI and max-pain distance features. |
| Ni, S. X., Pearson, N. D. and Poteshman, A. M. (2005). Stock price clustering on option expiration dates. *Journal of Financial Economics*, 78(1), 49 to 87. doi:10.1016/j.jfineco.2004.08.005 | Verified | Empirical pinning of US stocks at strikes on expiry, linked to hedging. |
| Golez, B. and Jackwerth, J. C. (2012). Pinning in the S&P 500 futures. *Journal of Financial Economics*, 106(3), 566 to 585. doi:10.1016/j.jfineco.2012.06.010 | Verified | Pinning at the index level, the right analogue for Nifty (an index product). |
| Ni, S. X., Pearson, N. D., Poteshman, A. M. and White, J. (2021). Does option trading have a pervasive impact on underlying stock prices? *Review of Financial Studies*, 34(4), 1952 to 1986. doi:10.1093/rfs/hhaa082 | Verified | Key precedent: market makers' net gamma predicts the size of underlying returns. Negative net gamma raises volatility, positive lowers it. This is the sign of the effect built into our simulator. |
| Baltussen, G., Da, Z., Lammers, S. and Martens, M. (2021). Hedging demand and market intraday momentum. *Journal of Financial Economics*, 142(1), 377 to 403. doi:10.1016/j.jfineco.2021.04.029 | Verified | Gamma hedging explains end-of-day momentum across many futures markets. Motivates the last-30-minute momentum test. |
| Barbon, A. and Buraschi, A. (2020). Gamma fragility. Working paper. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3725454 | Verified as working paper | Aggregate dealer gamma imbalance drives momentum or reversal and fragility. |
| Dim, C., Eraker, B. and Vilkov, G. (2024). 0DTEs: Trading, gamma risk and volatility propagation. Working paper. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4692190 | Verified as working paper | Important counterweight: finds *no* evidence that 0DTE gamma hedging amplifies S&P 500 volatility. Our design must be able to return a null, and the power study (E4) checks that it can. |
| Bandi, F. M., Fusari, N. and Renò, R. (2023). 0DTE option pricing. Working paper. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4503344 | Verified as working paper | Pricing very short-dated options. Relevant because Nifty expiry-day options are effectively 0DTE. |
| Brogaard, J., Han, J. and Won, P. Y. (2023). Does 0DTE options trading increase volatility? Working paper, SSRN abstract 4426358, first posted 26 April 2023. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4426358 | Verified from listings (year added 3 November 2026; no journal version found) | Another test of whether very short-dated options raise index volatility. |

**Gap.** The search found no peer-reviewed study of dealer gamma or market-maker hedging effects on Nifty or any NSE options. That gap is real, as far as a careful search can tell, and it is what this project fills. It also means there is no published Indian estimate of the effect size, which is why the power study uses a range of effect sizes rather than one assumed value.

## 4. Indian positioning and volatility

| Reference | Status | Relation to this project |
| --- | --- | --- |
| Sharma, V. K., Bhatia, S. and Roy, H. (2023). Investment behavior of foreign institutional investors and implied volatility dynamics. *Journal of Risk and Financial Management*, 16(11), 470. doi:10.3390/jrfm16110470 | Verified | FII derivative positions Granger-cause India VIX. Supports using participant-wise OI as a positioning signal. |
| Singh, L. G. H. and Chaudhary, A. K. (2024). Option volume and open interest for predicting underlying return. In *Financial Markets and Corporate Finance (ICFMCF 2023)*, Springer, pp. 103 to 118. doi:10.1007/978-981-97-6242-2_6 | Verified | A null result for OI ratios predicting Nifty *returns*. We forecast *volatility*, not direction, which is where positioning theory makes a sharper prediction. |
| Sen, R. and Mehrotra, P. (2016). Modeling jumps and volatility of the Indian stock market using high-frequency data. *Journal of Quantitative Economics*, 14(1), 137 to 150. doi:10.1007/s40953-016-0028-5 | Verified | Jumps in 5-minute Nifty data. Supports keeping a jump measure (bipower variation) in the feature set. |
| Mahajan, V., Thakan, S. and Malik, A. (2022). Modeling and forecasting the volatility of NIFTY 50 using GARCH and RNN models. *Economies*, 10(5), 102. doi:10.3390/economies10050102 | Verified | Daily-frequency evidence that GARCH slightly beats an LSTM on Nifty. A reason to require every ML model to beat the econometric baselines, and to not assume a sequence model will help. |
| SEBI (2024). Measures to strengthen equity index derivatives framework. Circular SEBI/HO/MRD/TPD-1/P/CIR/2024/132, 1 October 2024 | Verified | Source of the 20 Nov 2024 single-weekly-expiry rule and the extra 2% expiry-day margin on short options. Defines a regime boundary. |
| SEBI (2024). Press Release PR No. 37/2024, 23 September 2024 | Verified | 93% of individual F&O traders lost money in FY22 to FY24. Context for retail being net option buyers, which is why the US sign convention may be wrong for India. |
| NSE (2025). Circular No. 108/2025 (NSE/FAOP/68685), 23 June 2025 | Verified | Nifty weekly expiry moved to Tuesday for contracts expiring from 1 Sep 2025; first Tuesday expiry 2 Sep 2025. A Monday move announced in March 2025 (Circular 33/2025) was deferred and never took effect. |

## 5. Volatility models and realized measures

| Reference | Status | Relation to this project |
| --- | --- | --- |
| Engle, R. F. and Sokalska, M. E. (2012). Forecasting intraday volatility in the US equity market: multiplicative component GARCH. *Journal of Financial Econometrics*, 10(1), 54 to 83. doi:10.1093/jjfinec/nbr005 | Verified | The mcsGARCH model, used by the anchor paper and rebuilt in `niftygex.volatility`. |
| Andersen, T. G. and Bollerslev, T. (1997). Intraday periodicity and volatility persistence in financial markets. *Journal of Empirical Finance*, 4(2 to 3), 115 to 158. doi:10.1016/S0927-5398(97)00004-2 | Verified | Why the diurnal pattern must be removed before modelling intraday volatility dynamics. |
| Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics*, 7(2), 174 to 196. doi:10.1093/jjfinec/nbp001 | Verified | The HAR model, one rung of our forecast ladder. |
| Rogers, L. C. G. and Satchell, S. E. (1991). Estimating variance from high, low and closing prices. *Annals of Applied Probability*, 1(4), 504 to 512. doi:10.1214/aoap/1177005835 | Verified | The per-minute volatility measure the anchor paper tests. |
| Rogers, L. C. G., Satchell, S. E. and Yoon, Y. (1994). Estimating the volatility of stock prices: a comparison of methods that use high and low prices. *Applied Financial Economics*, 4(3), 241 to 247. doi:10.1080/758526905 | Verified | The reference the anchor paper cites for its volatility measure. |
| Barndorff-Nielsen, O. E. and Shephard, N. (2004). Power and bipower variation with stochastic volatility and jumps. *Journal of Financial Econometrics*, 2(1), 1 to 37. doi:10.1093/jjfinec/nbh001 | Verified | Jump-robust bipower variation, used for the jump-share feature. |
| Black, F. (1976). The pricing of commodity contracts. *Journal of Financial Economics*, 3(1 to 2), 167 to 179. doi:10.1016/0304-405X(76)90024-6 | Verified | Black-76, the pricing model for the greeks engine. |
| Easley, D., López de Prado, M. M. and O'Hara, M. (2012). Flow toxicity and liquidity in a high-frequency world. *Review of Financial Studies*, 25(5), 1457 to 1493. doi:10.1093/rfs/hhs053 | Verified | Bulk volume classification, our order-flow proxy when trade signs are missing. |

## 6. Connectedness and inference

| Reference | Status | Relation to this project |
| --- | --- | --- |
| Diebold, F. X. and Yilmaz, K. (2009). Measuring financial asset return and volatility spillovers. *Economic Journal*, 119(534), 158 to 171. doi:10.1111/j.1468-0297.2008.02208.x | Verified | The original spillover index (ordering-dependent). |
| Diebold, F. X. and Yilmaz, K. (2012). Better to give than to receive. *International Journal of Forecasting*, 28(1), 57 to 66. doi:10.1016/j.ijforecast.2011.02.006 | Verified | The connectedness measure the anchor paper uses. Defines the total index as the off-diagonal sum divided by N, which matters for reading the paper's tables (see the deep dive). |
| Pesaran, H. H. and Shin, Y. (1998). Generalized impulse response analysis in linear multivariate models. *Economics Letters*, 58(1), 17 to 29. doi:10.1016/S0165-1765(97)00214-0 | Partly verified (DOI built from the publisher ID, not seen printed) | The generalized variance decomposition behind DY 2012. |
| Newey, W. K. and West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703 to 708. doi:10.2307/1913610 | Verified | HAC standard errors, our alternative to minute-level Wilcoxon tests. |
| Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. *Annals of Statistics*, 17(3), 1217 to 1241. doi:10.1214/aos/1176347265 | Verified | Moving block bootstrap, used for connectedness confidence intervals. |
| Politis, D. N. and Romano, J. P. (1994). The stationary bootstrap. *JASA*, 89(428), 1303 to 1313. doi:10.1080/01621459.1994.10476870 | Verified | Alternative block bootstrap. |

## 7. Forecast evaluation and backtest hygiene

| Reference | Status | Relation to this project |
| --- | --- | --- |
| Patton, A. J. (2011). Volatility forecast comparison using imperfect volatility proxies. *Journal of Econometrics*, 160(1), 246 to 256. doi:10.1016/j.jeconom.2010.03.034 | Verified | Why we score with QLIKE and MSE only. |
| Diebold, F. X. and Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business and Economic Statistics*, 13(3), 253 to 263 | Partly verified (journal DOI not confirmed) | The test behind the GEX ablation. |
| Hansen, P. R., Lunde, A. and Nason, J. M. (2011). The model confidence set. *Econometrica*, 79(2), 453 to 497. doi:10.3982/ECTA5771 | Verified | Compares the whole model ladder at once. |
| Bailey, D. H. and López de Prado, M. (2014). The deflated Sharpe ratio. *Journal of Portfolio Management*, 40(5), 94 to 107. doi:10.3905/jpm.2014.40.5.094 | Verified | Required correction if the trading layer is extended (see alternatives). |
| Gao, L., Han, Y., Li, S. Z. and Zhou, G. (2018). Market intraday momentum. *Journal of Financial Economics*, 129(2), 394 to 414. doi:10.1016/j.jfineco.2018.05.009 | Verified | Baseline intraday momentum result that the GEX momentum test conditions on. |

## 8. Methods references added for the working paper (checked 3 November 2026)

These references were added when the methods were written up (multiple testing, robust inference, forecast comparison, noise-robust volatility, synthetic control). An AI agent checked each one on 3 November 2026 against publisher, RePEc/IDEAS, JSTOR, Project Euclid, SSRN and NeurIPS listings as quoted by a web search tool. Crossref and the publisher pages themselves could not be opened from our setup, so "verified from listings" is one step short of the standard used in sections 1 to 7; a re-check against Crossref is the last step before any submission.

| Reference | Status |
| --- | --- |
| Abadie, A., Diamond, A. and Hainmueller, J. (2010). Synthetic control methods for comparative case studies: Estimating the effect of California's tobacco control program. *Journal of the American Statistical Association*, 105(490), 493 to 505. doi:10.1198/jasa.2009.ap08746 | Verified from listings |
| Barndorff-Nielsen, O. E., Hansen, P. R., Lunde, A. and Shephard, N. (2008). Designing realized kernels to measure the ex post variation of equity prices in the presence of noise. *Econometrica*, 76(6), 1481 to 1536. doi:10.3982/ECTA6495 | Verified from listings |
| Benjamini, Y. and Hochberg, Y. (1995). Controlling the false discovery rate: A practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society, Series B*, 57(1), 289 to 300. doi:10.1111/j.2517-6161.1995.tb02031.x | Verified from listings |
| Benjamini, Y. and Yekutieli, D. (2001). The control of the false discovery rate in multiple testing under dependency. *Annals of Statistics*, 29(4), 1165 to 1188. doi:10.1214/aos/1013699998 | Verified from listings |
| Bogousslavsky, V. and Muravyev, D. (2023). Who trades at the close? Implications for price discovery and liquidity. *Journal of Financial Markets*, 66, 100852. doi:10.1016/j.finmar.2023.100852 | Verified from listings |
| Bollen, N. P. B. and Whaley, R. E. (2004). Does net buying pressure affect the shape of implied volatility functions? *Journal of Finance*, 59(2), 711 to 753. doi:10.1111/j.1540-6261.2004.00647.x | Verified from listings |
| Cameron, A. C., Gelbach, J. B. and Miller, D. L. (2008). Bootstrap-based improvements for inference with clustered errors. *Review of Economics and Statistics*, 90(3), 414 to 427. doi:10.1162/rest.90.3.414 | Verified from listings |
| Conley, T. G. and Taber, C. R. (2011). Inference with "difference in differences" with a small number of policy changes. *Review of Economics and Statistics*, 93(1), 113 to 125. doi:10.1162/REST_a_00049 | Verified from listings; middle initial of Taber not shown in sources |
| Driscoll, J. C. and Kraay, A. C. (1998). Consistent covariance matrix estimation with spatially dependent panel data. *Review of Economics and Statistics*, 80(4), 549 to 560. doi:10.1162/003465398557825 | Verified from listings |
| Gârleanu, N., Pedersen, L. H. and Poteshman, A. M. (2009). Demand-based option pricing. *Review of Financial Studies*, 22(10), 4259 to 4299. doi:10.1093/rfs/hhp005 | Corrected (name diacritic), verified from listings |
| Giacomini, R. and White, H. (2006). Tests of conditional predictive ability. *Econometrica*, 74(6), 1545 to 1578. doi:10.1111/j.1468-0262.2006.00718.x | Verified from listings |
| Hansen, P. R. (2005). A test for superior predictive ability. *Journal of Business and Economic Statistics*, 23(4), 365 to 380. doi:10.1198/073500105000000063 | Verified from listings |
| Harvey, D., Leybourne, S. and Newbold, P. (1997). Testing the equality of prediction mean squared errors. *International Journal of Forecasting*, 13(2), 281 to 291. doi:10.1016/S0169-2070(96)00719-4 | Verified from listings |
| Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65 to 70. https://www.jstor.org/stable/4615733 | Verified from listings; no DOI confirmed |
| Jacod, J., Li, Y., Mykland, P. A., Podolskij, M. and Vetter, M. (2009). Microstructure noise in the continuous case: The pre-averaging approach. *Stochastic Processes and their Applications*, 119(7), 2249 to 2276. doi:10.1016/j.spa.2008.11.004 | Verified from listings |
| Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q. and Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. In *Advances in Neural Information Processing Systems 30* (NIPS 2017), pp. 3146 to 3154 | Verified from listings |
| Kiefer, N. M. and Vogelsang, T. J. (2005). A new asymptotic theory for heteroskedasticity-autocorrelation robust tests. *Econometric Theory*, 21(6), 1130 to 1164. doi:10.1017/S0266466605050565 | Verified from listings; DOI from one catalogue record |
| MacKinnon, J. G. and Webb, M. D. (2017). Wild bootstrap inference for wildly different cluster sizes. *Journal of Applied Econometrics*, 32(2), 233 to 254. doi:10.1002/jae.2508 | Verified from listings |
| Nelson, D. B. (1991). Conditional heteroskedasticity in asset returns: A new approach. *Econometrica*, 59(2), 347 to 370. doi:10.2307/2938260 | Verified from listings |
| Romano, J. P. and Wolf, M. (2005). Stepwise multiple testing as formalized data snooping. *Econometrica*, 73(4), 1237 to 1282. doi:10.1111/j.1468-0262.2005.00615.x | Verified from listings |
| White, H. (2000). A reality check for data snooping. *Econometrica*, 68(5), 1097 to 1126. doi:10.1111/1468-0262.00152 | Verified from listings |
| Zhang, L., Mykland, P. A. and Aït-Sahalia, Y. (2005). A tale of two time scales: Determining integrated volatility with noisy high-frequency data. *Journal of the American Statistical Association*, 100(472), 1394 to 1411. doi:10.1198/016214505000000169 | Verified from listings |
| Bollerslev, T., Hood, B., Huss, J. and Pedersen, L. H. (2018). Risk everywhere: Modeling and managing volatility. *Review of Financial Studies*, 31(7), 2729 to 2773. doi:10.1093/rfs/hhy041 | Verified from listings (used for P103, the economic value of the forecasts) |

## What the literature implies for the design

1. The mechanism has strong US support (Ni et al. 2021; Baltussen et al. 2021) and a credible null in the newest 0DTE work (Dim, Eraker and Vilkov 2024). So the honest framing is a test, not a confirmation, and the design must be able to produce a clean null.
2. The sign of dealer positioning is the central identification problem. US studies use exchange data that flags market-maker accounts. NSE publishes only aggregate participant-wise OI (Client, DII, FII, Pro) across all index options. Our design therefore reports every result under several sign rules and measures, by simulation, how much a noisy sign costs.
3. Indian ML evidence at daily frequency does not favour deep sequence models (Mahajan et al. 2022). The ladder therefore puts trees before sequence models and requires each rung to beat the one below out of sample.
4. Forecast comparison must use robust losses (Patton 2011) and day-level dependence-aware tests. The same dependence problem affects the anchor paper's minute-level Wilcoxon tests, which our experiment E2 quantifies.
