# Delhivery Last-Mile Delay Intelligence Pipeline

Capstone project for the AI and ML Certification Program, E&ICT Academy, IIT Roorkee. The project uses synthetic shipment data to estimate delivery hours, flag parcels likely to miss their promise, and group hubs into operating segments.

## Data findings

All numbers below come from `outputs/part_a_delay_by_group.csv` and `outputs/part_a_correlations.csv`. After removing duplicates there are 1,200 unique shipments, of which 278 are delayed (delay rate 0.2317).

1. **Service type.** Express parcels are delayed most often: 119 of 243 shipments (delay rate 0.4897). Standard is at 0.2056 and Economy at 0.0857. Express has the shortest promise, so it has the least room for error.

2. **Hub tier.** The delay rate rises steadily from Tier 1 to Tier 3: 0.0400 for Tier 1 (16 of 400), 0.2150 for Tier 2 (86 of 400) and 0.4400 for Tier 3 (176 of 400).

3. **Distance.** Longer lanes are delayed far more often: 0.0219 for 0-200 km, 0.1368 for 201-400 km, 0.4364 for 401-600 km and 0.8523 for 601+ km. Of the 88 parcels in the longest band, 75 were late.

4. **Strongest correlations with delivery_hours.** Ranked by absolute value, the top three columns are promised_hours (0.8372), distance_km (0.4013) and n_stops (0.3247). Longer promises, longer lanes and more stops on the rider's route all go with longer delivery times.

## Model results (regression)

All numbers are from `outputs/part_b_metrics.csv`, `part_b_tree_depth.csv`, `part_b_gb_learning_rate.csv` and `part_b_cv.csv`. Split: 840 train / 180 validation / 180 test (`random_state=42`).

**Baseline.** Predicting the training mean for every parcel gives a validation R² of -0.0212 (RMSE 18.8552 hours). An R² near zero means the average explains none of the variation in delivery time, so any real model must beat this.

**Validation winner.** Linear regression had the lowest validation RMSE, 4.8912 hours (R² 0.9313), just ahead of gradient boosting at 4.9165 hours (R² 0.9306). Random forest scored 5.5059 and the decision tree 6.2796.

**Tree depth.** Depth 1 is high bias: train RMSE 11.6334 and validation RMSE 12.1324 are both high and only 0.50 apart, so the tree is too simple. With no depth limit the tree is high variance: train RMSE is 0.0000 but validation RMSE is 7.8368, a gap of 7.84. Depth 12 shows the same problem (gap 5.99). The best depth was 6 (validation RMSE 6.2796).

**Learning rate.** For gradient boosting, 0.1 was best (validation R² 0.9306, RMSE 4.9165). At 0.01 the R² is only 0.7127 because each tree corrects too little, so 100 trees are not enough (too slow). At 1.0 the R² is 0.8440 because each tree overshoots the correction and the model becomes unstable.

**Cross-validation.** On the 1,020 train and validation rows with one shared 5-fold split, gradient boosting had a mean R² of 0.9353 (std 0.0045) and linear regression 0.9261 (std 0.0048). This disagrees with the validation ranking. 

**Which model I would ship.** I would ship linear regression. It had the lowest validation RMSE (4.8912 vs 4.9165 hours) and it is the model I scored on the test rows (R² 0.9493, RMSE 4.5235 hours), so its test figure is an honest estimate. Gradient boosting had a slightly higher cross-validated R² (0.9353 vs 0.9261), but the gap is small, in hours it is a fraction of an hour, and linear regression is simpler to explain to planners and faster to run. If planners later need the last bit of accuracy, gradient boosting would be the first alternative to test.

**Final test.** Linear regression, the validation winner, was scored once on the test rows: R² 0.9493, RMSE 4.5235 hours, MAE 3.5671 hours.