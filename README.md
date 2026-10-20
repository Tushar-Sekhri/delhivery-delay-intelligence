# Delhivery Last-Mile Delay Intelligence Pipeline

Capstone project for the AI and ML Certification Program, E&ICT Academy, IIT Roorkee. The project uses synthetic shipment data to estimate delivery hours, flag parcels likely to miss their promise, and group hubs into operating segments.

## Data findings

All numbers below come from `outputs/part_a_delay_by_group.csv` and `outputs/part_a_correlations.csv`. After removing duplicates there are 1,200 unique shipments, of which 278 are delayed (delay rate 0.2317).

1. **Service type.** Express parcels are delayed most often: 119 of 243 shipments (delay rate 0.4897). Standard is at 0.2056 and Economy at 0.0857. Express has the shortest promise, so it has the least room for error.

2. **Hub tier.** The delay rate rises steadily from Tier 1 to Tier 3: 0.0400 for Tier 1 (16 of 400), 0.2150 for Tier 2 (86 of 400) and 0.4400 for Tier 3 (176 of 400).

3. **Distance.** Longer lanes are delayed far more often: 0.0219 for 0-200 km, 0.1368 for 201-400 km, 0.4364 for 401-600 km and 0.8523 for 601+ km. Of the 88 parcels in the longest band, 75 were late.

4. **Strongest correlations with delivery_hours.** Ranked by absolute value, the top three columns are promised_hours (0.8372), distance_km (0.4013) and n_stops (0.3247). Longer promises, longer lanes and more stops on the rider's route all go with longer delivery times.
