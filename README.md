# sm_system
Virtual simulation environment for identifying systems with 'SM' structure.

An SM system is a class of nonlinear systems introduced by Baumgartner and Rugh [BR75]. It is
formalized as a dynamic polynomial of order M and can be synthesized as M parallel branches. Each branch Hm consists of a linear pre-filter Hm1 a static nonlinearity (·)^m and a linear post-filter
Hm2.

![SM-System](https://github.com/2mrwolke/sm_system/blob/main/SM.png)

[BR75]
Stephen Baumgartner and Wilson Rugh. “Complete identification of a class of nonlinear sys-
tems from steady-state frequency response.” In: IEEE Transactions on Circuits and Systems
22.9 (1975), pp. 753–759.
