NB. ============================================================
NB. ad.ijs -- reverse-mode AD over a closed set of J primitives (v4)
NB.
NB. Conventions:
NB.   - node ref = plain INT id; literal values live in leaves
NB.   - tape row = box of (id;value;op;parents) ; parents = INT 2-list
NB.   - binary ops dyadic (x op y), unary ops monadic (op y)
NB.   - ADSETP takes boxed (name;value) pairs -- no J-side eval
NB.   - adshrink undoes J broadcast in VJPs (parent value vs grad)
NB. J9.8-beta pitfalls honored: no top-level control words anywhere;
NB. ids forced INT via (2#0)+... ; stdlib 'empty' never used.
NB. v4 additions: nrsumr (arbitrary-axis sum), nrk (rank-dispatched
NB. dyadic arithmetic), nlt (compare mask), nwhere (mask*a+(1-mask)*b).
NB. ============================================================
coclass 'z'

ADT=: 0$<''
ADN=: 0
G=: ''
ADLEAVES=: ''
adleafids=: ''

nnew=: 3 : 0
 'v o p'=. y
 i=. ADN
 ADN=: >:i
 ADT=: ADT , <(i;(>v);(>o);<p)
 i
)

nrow=: 3 : '> y { ADT'
nval=: 3 : '> 1 { nrow y'
nop =: 3 : '> 2 { nrow y'
npar=: 3 : '> 3 { nrow y'

adV=: 3 : 'nval y'
adR=: 3 : '1'

adP=: 4 : '(2#0) + x,y'
adP1=: 3 : 0
 y , y
)

nleaf=: 3 : 0
 nnew (<y) , (<'leaf') , <((2#0)+0 0)
)

NB. ---------- op constructors ----------
nadd=: 4 : 0
 nnew (<(adV x)+adV y) , (<'add') , <(x adP y)
)
nsub=: 4 : 0
 nnew (<(adV x)-adV y) , (<'sub') , <(x adP y)
)
nmul=: 4 : 0
 nnew (<(adV x)*adV y) , (<'mul') , <(x adP y)
)
ndiv=: 4 : 0
 nnew (<(adV x)%adV y) , (<'div') , <(x adP y)
)
nneg=: 3 : 0
 nnew (<-adV y) , (<'neg') , <(adP1 y)
)
nsq=: 3 : 0
 nnew (<*: adV y) , (<'sq') , <(adP1 y)
)
nexp=: 3 : 0
 nnew (<^ adV y) , (<'exp') , <(adP1 y)
)
nlog=: 3 : 0
 nnew (<^. adV y) , (<'log') , <(adP1 y)
)
ntanh=: 3 : 0
 nnew (<7&o. adV y) , (<'tanh') , <(adP1 y)
)
npow=: 4 : 0
 nnew (<(adV x)^ y) , (<'pow') , <(x,x)
)
nsum=: 3 : 0
 nnew (<+/ adV y) , (<'sum') , <(adP1 y)
)
nmax=: 3 : 0
 nnew (<>/ adV y) , (<'max') , <(adP1 y)
)
nmp=: 4 : 0
 nnew (<(adV x) mp adV y) , (<'mp') , <(x adP y)
)
nbadd=: 4 : 0
 nnew (<(adV x) +"(1) adV y) , (<'badd') , <(x adP y)
)
ntr=: 3 : 0
 nnew (<|: adV y) , (<'tr') , <(adP1 y)
)
nemax=: 4 : 0
 nnew (<(adV x) >. adV y) , (<'emax') , <(x adP y)
)
nrsub1=: 4 : 0
 nnew (<(adV x) -"(1 0) adV y) , (<'rsub1') , <(x adP y)
)
nrsum1=: 3 : 0
 nnew (<+/"1 adV y) , (<'rsum1') , <(adP1 y)
)
nrdiv1=: 4 : 0
 nnew (<(adV x) %"(1 0) adV y) , (<'rdiv1') , <(x adP y)
)
nrmax1=: 3 : 0
 nnew (< >./"1 adV y) , (<'rmax1') , <(adP1 y)
)
nreshape=: 4 : 0
 nnew (<(adV y) $ , adV x) , (<'reshape') , <(x adP y)
)

NB. take/drop/gather: x = data node, y = INT spec node (constant leaf:
NB. take/drop spec = count vector; gather spec = index vector)
ntake=: 4 : 0
 nnew (<((adV y) {."1 adV x)) , (<'take') , <(x adP y)
)
ndrop=: 4 : 0
 nnew (<((adV y) }."1 adV x)) , (<'drop') , <(x adP y)
)
ngather=: 4 : 0
 nnew (<((,adV y) { adV x)) , (<'gather') , <(x adP y)
)

NB. axis-sum: sum over axis k (0-based, from left) of x. y = INT-k constant
NB. leaf. Forward rotates axis k to the end, +/"1; result keeps the rotated
NB. order of the other axes.
nrsumr=: 4 : 0
 nnew (<(((+/) " 1) (((i. # $ adV x) -. adV y) , adV y) |: adV x)) , (<'rsumr') , <(x adP y)
)

NB. compare mask: B01 result, gradient 0 (use as gate multiplier)
nlt=: 4 : 0
 nnew (<(adV x) < adV y) , (<'lt') , <(x adP y)
)

NB. gated mix: x*(m) + y*(1-m) where m is the mask node. Differentiable
NB. through both x and y; m itself gets no gradient (B01).
nwhere=: 4 : 0
 NB. y = (mask node; b node) boxed pair? -- tape parents are a 2-list, so
 NB. nwhere takes mask as x and builds two sub-nodes via caller helpers.
 NB. Simpler: single primitive with mask x and value y: mask * y.
 nnew (<((adV x) * adV y)) , (<'where') , <(x adP y)
)

mp=: +/ . *

NB. ---------- broadcast shrink ----------
adshrink=: 4 : 0
 NB. x = parent NODE ID, y = grad; shrink grad to parent shape
 pa=. $ adV x
 ga=. $ y
 if. pa -: ga do. y return. end.
 if. (0=#pa) do. (+/,y) return. end.
 if. ((1=#pa)*.(1<#ga)) do. +/"1 y return. end.
 if. ((1<#pa)*.(1=#ga)) do. (pa$y) return. end.
 y
)

NB. ---------- backward ----------
acc=: 4 : 0
 G=: (<(>x{G) + y) x} G
)

nback=: 3 : 0
 G=: ADN $ <0
 G=: (<1) y} G
 i=. <:ADN
 while. i >: 0 do.
  g=. >i{G
  if. #g do.
   o=. nop i
   p=. npar i
   select. o
   case. 'leaf' do.
   case. 'add' do.
    'a b'=. p
    if. adR a do. a acc (a adshrink g) end.
    if. adR b do. b acc (b adshrink g) end.
   case. 'sub' do.
    'a b'=. p
    if. adR a do. a acc (a adshrink g) end.
    if. adR b do. b acc -(b adshrink g) end.
   case. 'neg' do.
    'a b'=. p
    if. adR a do. a acc -g end.
   case. 'mul' do.
    'a b'=. p
    if. adR a do. a acc (a adshrink (g*adV b)) end.
    if. adR b do. b acc (b adshrink (g*adV a)) end.
   case. 'div' do.
    'a b'=. p
    if. adR a do. a acc (a adshrink (g%adV b)) end.
    if. adR b do. b acc (b adshrink ((-g) * (adV a) % (adV b)*(adV b))) end.
   case. 'sq' do.
    'a b'=. p
    if. adR a do. a acc g*2*adV a end.
   case. 'pow' do.
    'a b'=. p
    if. adR a do. a acc g*(adV b)*adV a^(adV b)-1 end.
   case. 'exp' do.
    'a b'=. p
    if. adR a do. a acc g*^ adV a end.
   case. 'log' do.
    'a b'=. p
    if. adR a do. a acc g%adV a end.
   case. 'tanh' do.
    'a b'=. p
    if. adR a do. a acc g*(1 - *: 7&o. adV a) end.
   case. 'sum' do.
    'a b'=. p
    if. adR a do. a acc (($adV a) $ g) end.
   case. 'max' do.
    'a b'=. p
    if. adR a do. a acc (adV a) = >./ adV a end.
   case. 'badd' do.
    'a b'=. p
    if. adR a do. a acc g end.
    if. adR b do. b acc +/ g end.
   case. 'mp' do.
    'a b'=. p
    if. adR a do. a acc g mp |: adV b end.
    if. adR b do. b acc (|: adV a) mp g end.
   case. 'tr' do.
    'a b'=. p
    if. adR a do. a acc |: g end.
   case. 'emax' do.
    'a b'=. p
    if. adR a do. a acc g * (adV a) >: adV b end.
    if. adR b do. b acc g * (adV b) >: adV a end.
   case. 'rsub1' do.
    'a b'=. p
    if. adR a do. a acc g end.
    if. adR b do. b acc - +/"1 g end.
   case. 'rsum1' do.
    'a b'=. p
    NB. per-row sums: replicate each grad element across its row. NB. $ keeps
    NB. the item shape of an array argument -- ravel first or the result
    NB. gains a trailing axis!
    if. adR a do.
     pa2=. $ adV a
     eg=. ({: pa2) # , g          NB. replicate each row-grad element across its row
     a acc ((pa2 $ eg))
    end.
   case. 'rdiv1' do.
    'a b'=. p
    if. adR a do. a acc g %"(1 0) adV b end.
    if. adR b do. b acc (- (+/"1 (g * (adV a) %"(1 0) ((adV b) * adV b)))) end.
   case. 'rmax1' do.
    'a b'=. p
    if. adR a do. a acc g * ((adV a) ="(1 0) nval i) end.   NB. g * one-hot mask
   case. 'reshape' do.
    'a b'=. p
    if. adR a do. a acc (($adV a) $ ,g) end.
   case. 'take' do.
    'a b'=. p
    NB. forward: trailing axis truncated to n=spec; VJP: g with n zero
    NB. columns appended (drop's complement)
    if. adR a do.
     n=. {: adV b
     shp=. $adV a
     tz=. ((0 { shp) , n) $ 0.0
     a acc (g ,"1 tz)
    end.
   case. 'drop' do.
    'a b'=. p
    NB. forward: trailing axis dropped by n; VJP: n zero columns prepended
    if. adR a do.
     n=. {: adV b
     shp=. $adV a
     pz=. ((0 { shp) , n) $ 0.0
     a acc (pz ,"1 g)
    end.
   case. 'gather' do.
    'a b'=. p
    NB. scatter-ACCUMULATE: dup indices must sum. Build (n x k) equality
    NB. table S=(i.n)="(0 1) idx; grad = S mp g. (amend } would replace!)
    if. adR a do.
     nrn=. 0 { $ adV a
     a acc (((i. nrn) ="(0 1) , adV b) mp g)
    end.
   case. 'rsumr' do.
    'a b'=. p
    NB. forward rotated axis k to the end (pv = axes-without-k , k) and
    NB. reduced the trailing axis. g is in rotated frame order (leading
    NB. n-1 axes of the rotated array). VJP: replicate each frame scalar
    NB. across the reduced axis (ravel-replicate, exactly the rsum1 trick),
    NB. reshape to the ROTATED input shape, then un-rotate with (/: pv).
    if. adR a do.
     ash=: $ adV a
     kx=. adV b
     pv=. ((i. # ash) -. kx) , kx
     rsh=. ash { ~ pv                      NB. rotated input shape
     eg=. ({: rsh) # , g
     a acc (((/: pv) |: (rsh $ eg)))
    end.
   case. 'lt' do.
    'a b'=. p
    NB. B01 mask: no gradient flows (non-differentiable boundary)
   case. 'where' do.
    'a b'=. p
    NB. mask * y: same VJP as mul with the mask treated as constant b
    if. adR b do. b acc (b adshrink (g * adV a)) end.
   end.
  end.
  i=. <:i
 end.
)

NB. ---------- drivers ----------
ADSET=: 3 : 0
 NB. y: boxed list of NAMES (values read by nleaf (". nm) at caller's risk)
 ADT=: 0$<''
 ADN=: 0
 ADLEAVES=: ''
 adleafids=: ''
 for_n. y do.
  nm=. >n
  id=. nleaf (". nm)
  adleafids=: adleafids , <nm;id
  ADLEAVES=: ADLEAVES , nm
 end.
 i=. 0
)

ADSETP=: 3 : 0
 NB. y: boxed list of (name;value) pairs -- value-passing, no eval
 ADT=: 0$<''
 ADN=: 0
 ADLEAVES=: ''
 adleafids=: ''
 for_p. y do.
  'nm v'=. >p
  ADADDLEAF nm;v
 end.
 i=. 0
)

ADADDLEAF=: 3 : 0
 NB. y: (name;value) pair -- append one leaf without clearing the tape
 'nm v'=. y
 id=. nleaf v
 adleafids=: adleafids , <nm;id
 ADLEAVES=: ADLEAVES , nm
 i=. 0
)

ADSETVALID=: 3 : 0
 NB. y: (nodeid;value) -- overwrite a node's stored value (for refreshing leaf inputs)
 'id v'=. y
 row=: >id{ADT
 ADT=: (<(<id;v;(>2{row);(>3{row))) id} ADT
 i=. 0
)

ADGET=: 3 : 0
 nback y
 r=. ''
 for_l. adleafids do.
  'nm id'=. >l
  r=. r , < > id { G
 end.
 r
)
