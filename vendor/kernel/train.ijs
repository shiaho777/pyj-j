NB. ============================================================
NB. host/train.ijs -- two-spiral classifier trained entirely in J.
NB.
NB. Driven by the C host (host/train.c) through the 5-call ABI:
NB.   JDo "0!:0 <'ad.ijs'"       (host loads the AD core first)
NB.   JDo "0!:0 <'train.ijs'"    load this script (defines sp_ verbs)
NB.   JDo "sp_data 384"          build the two-spiral dataset
NB.   JDo "sp_build 0"           init params
NB.   JDo "sp_train 1500"        GD steps, J-side, zero host round-trips
NB.   JGetM "sp_acc"             scalar: final train accuracy
NB.
NB. Per step the tape is rebuilt (ADSETP resets ADT/ADN), the forward
NB. sentences re-run, and GD updates the param globals. The tape is a
NB. fixed ~30-node graph; ids 0..6 are the leaves in ADSETP order:
NB. 0 SPX, 1 SPY1, 2 SPW1, 3 SPB2, 4 SPW2, 5 SPB3, 6 SPW3.
NB.
NB. Embedded-mode rules: no top-level control words; no stdlib; INT ids
NB. forced via (2#0)+...; `}` amend replaces (acc accumulates).
NB. ============================================================
coclass 'z'

NB. ---------- dataset: two interleaved spirals ----------
sp_seed=: 3 : 0
 9!:1 y
 i. 0
)

sp_data=: 3 : 0
 NB. y: points per class. Outputs:
 NB.   sp_x  (2n,2) FL inputs;  spy (2n,) INT labels;  spy1 (2n,2) FL one-hot
 n=. y
 t=. (%n) * i. n
 rr0=. t + 0.06 * (?n) - 0.5
 rr1=. t + 0.06 * (?n) - 0.5
 aa0=. (2 * o. 1) * (3 * t) + 0.06 * (?n) - 0.5
 aa1=. (2 * o. 1) * (3 * t) + 0.06 * (?n) - 0.5
 px=. (rr0 * 1 o. aa0) ,. rr0 * 2 o. aa0
 py=. (rr1 * 1 o. aa1) ,. rr1 * 2 o. aa1
 sp_x=: px , py
 spy=: (n $ 0) , (n $ 1)
 spy1=: (spy =/ 0 1) + 0
 i. 0
)

NB. ---------- params ----------
sp_build=: 3 : 0
 spw1=: (?2 16 $ 0) - 0.5
 spb2=: ?16 $ 0
 spw2=: (?16 16 $ 0) - 0.5
 spb3=: ?2 $ 0
 spw3=: (?16 2 $ 0) - 0.5
 splr=: 0.1
 i. 0
)

NB. ---------- one training step; returns mean CE ----------
NB. leaves in ADSETP order: 0 SPX 1 SPY1 2 SPW1 3 SPB2 4 SPW2 5 SPB3 6 SPW3
sp_step=: 3 : 0
 ADSETP (<'SPX';sp_x) , (<'SPY1';spy1) , (<'SPW1';spw1) , (<'SPB2';spb2) , (<'SPW2';spw2) , (<'SPB3';spb3) , (<'SPW3';spw3)
 u1=: 0 nmp 2              NB. SPX  mp SPW1  (2n,16)
 h1=: ntanh (u1 nbadd 3)   NB. tanh(u1 + SPB2)
 u2=: h1 nmp 4             NB. h1   mp SPW2
 h2=: ntanh u2             NB. (2n,16)
 NB. third layer: logits = h2 mp SPW3 (2n,2); softmax CE, mean over rows:
 lg=: h2 nmp 6             NB. logits (2n,2)
 m=: nrmax1 lg             NB. rowmax
 e=: nexp (lg nrsub1 m)    NB. exp(logits - rowmax)
 s=: nrsum1 e              NB. row sums (2n,)
 lse=: m nadd (nlog s)     NB. logsumexp per row
 diff=: lg nrsub1 lse      NB. log softmax per row
 NB. SUM cross-entropy over rows (mean handled by lr scaling in sp_train):
 ce=: nsum (diff nmul 1)
 ce=: nneg ce
 i. 0
)

NB. One GD step: rebuild tape, forward, backprop, update params.
sp_step1=: 3 : 0
 sp_step 0
 gg=: ADGET ce
 lrn=. splr % # spy       NB. SUM-loss grads -> mean-scale step
 spw1=: spw1 - lrn * >2{gg
 spb2=: spb2 - lrn * >3{gg
 spw2=: spw2 - lrn * >4{gg
 spb3=: spb3 - lrn * >5{gg
 spw3=: spw3 - lrn * >6{gg
 i. 0
)

sp_train=: 3 : 0
 NB. y: number of GD steps, run inside this single JDo.
 for_k. i. y do. sp_step1 0 end.
 i. 0
)

NB. ---------- accuracy: argmax logits == label (2 classes) ----------
NB. Name-collision warning: ad.ijs defines the VERB acc (the gradient
NB. accumulator used by nback). Nothing here may assign a global named
NB. acc -- a noun by that name breaks nback's parse on its next call.
NB. All trainer state stays sp_-prefixed.
sp_acc=: 3 : 0
 sp_step 0
 sp_lgv=. adV lg
 NB. predicted class = 1 iff logit[1] > logit[0]; correct = pred = label.
 sp_pred=. (1 {"1 sp_lgv) > 0 {"1 sp_lgv
 sp_hit=. sp_pred = spy
 spacc=: (+/ sp_hit) % # spy
 spc=: +/ sp_hit
 i. 0
)

NB. ---------- inference probe (vendor demo) ----------
NB. sp_probe y: y is a 2-element FL list (one point). Runs the trained
NB. forward pass on the single row and stores logits in sp_plogits (2,).
NB. Uses the plain J weights (not tape ops) -- inference needs no gradient.
sp_probe=: 3 : 0
 NB. row pipeline: u1 = y mp spw1 (16,); h1 = tanh(u1+b2); u2 = h1 mp spw2;
 NB. h2 = tanh u2; logits = h2 mp spw3 (2,).
 u1p=. (,y) mp spw1
 h1p=. 7&o. (u1p + spb2)
 u2p=. h1p mp spw2
 h2p=. 7&o. u2p
 sp_plogits=: h2p mp spw3
 i. 0
)
