The operational efficacy of drone-assisted attended delivery is significantly undermined by service
failures stemming from customer presence uncertainty. This paper addresses this challenge by formulating a
dynamic vehicle routing problem that necessitates immediate re-planning upon such disruptions. We introduce
the Risk-Aware, Multi-Visit Drone-Assisted Pickup and Delivery Problem with Uncertain Customer Presence
(RA-MV-DAPDPUCP) and propose a corresponding mixed-integer non-linear programming (MINLP) model.
The model uniquely captures the problem’s complexities by integrating multi-visit, synchronized pickup and
delivery, and the cascading effects of service failures. To overcome the inherent computational complexity, we
develop a two-stage solution framework consisting of a region-based algorithm for initial routing and an online
re-planning mechanism. The re-planning is driven by an Adaptive Large Neighborhood Search (ALNS) enhanced
with a pheromone learning mechanism. Computational experiments, performed on a new set of benchmark
instances generated for this problem, demonstrate the superiority of our proposed ALNS, which outperforms
the commercial solver Gurobi by an average of 1.17% in solution quality. Furthermore, the results validate the
effectiveness of the collaborative truck-drone system, particularly under uncertain customer presence (UCP).
Compared to a traditional truck-only system, our approach reduces total operational costs by up to 14.39%, with
benefits being most prominent in high-penalty scenarios. Finally, extensive sensitivity analyses on key operational
parameters yield actionable managerial insights for practitioners.
