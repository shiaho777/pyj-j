NB. ============================================================
NB. adcomp.ijs -- tape-to-J-verb compiler
NB.
NB. Reads the AD tape built by ad.ijs and emits the source of
NB. two explicit J verbs:
NB.   adcF<n>  forward only (result = loss value)
NB.   adcB<n>  fused forward+backward; writes adcgL (loss value)
NB.            and adcg0..adcg(L-1) (leaf grads, ADSETP order)
NB.
NB. Why "compiled": the interpreted path (ADGET) walks a boxed
NB. tape with select. dispatch and nval/nop/npar row reads per
NB. node, once per backward. The generated verb is straight-line
NB. code: locals v<i>/g<i>, no boxing, no dispatch -- the same
NB. thing a human would hand-write, produced mechanically from
NB. the tape. This is J's special-code idea applied to our AD.
NB.
NB. Leaf inputs: real leaves read globals by their ADSETP names
NB. ({name}_v), so a training step only pushes new values and
NB. calls adcB<n> ''. Constant leaves (created by nleaf outside
NB. ADSETP: numeric literals, reshape dims) are BAKED at compile
NB. time into adcc<i> globals.
NB.
NB. VJP expressions mirror ad.ijs nback exactly (including its
NB. quirks: 'max' drops the incoming g; 'pow' uses the parent
NB. value where the exponent should be). Parity with the
NB. interpreted path is the invariant, tested in test_compile.py.
NB. ============================================================
coclass 'z'

LF=: 10 { a.   NB. embedded mode has no stdlib; define our own

ilocate=: 4 : 0
 NB. first index of substring y in x; #x when absent or needle longer
 if. (#y) > (#x) do. #x return. end.
 (y E. x) i. 1
)

adcsubst=: 4 : 0
 NB. x = template, y = 2-item boxed (aName;bName); splice %A / %B
 'an bn'=. >y
 i=. x ilocate '%A'
 while. i < #x do.
  x=. ((i{.x) , an , ((i+2)}.x))
  i=. x ilocate '%A'
 end.
 j=. x ilocate '%B'
 while. j < #x do.
  x=. ((j{.x) , bn , ((j+2)}.x))
  j=. x ilocate '%B'
 end.
 x
)

NB. shrink grad y to the shape of parent VALUE x (undoes broadcast;
NB. same rules as ad.ijs adshrink but value-based instead of id-based)
adcshrink=: 4 : 0
 pa=. $x
 ga=. $y
 if. pa -: ga do. y return. end.
 if. 0 = #pa do. (+/ , y) return. end.
 if. (1 = #pa) *. (1 < #ga) do. +/"1 y return. end.
 if. (1 < #pa) *. (1 = #ga) do. (pa $ y) return. end.
 y
)

NB. forward expression template per op; %A/%B = operand value names
adcfdesc=: 3 : 0
 select. y
 case. 'add' do. '(%A)+(%B)'
 case. 'sub' do. '(%A)-(%B)'
 case. 'mul' do. '(%A)*(%B)'
 case. 'div' do. '(%A) % (%B)'
 case. 'neg' do. '-(%A)'
 case. 'sq' do. '*:(%A)'
 case. 'exp' do. '^(%A)'
 case. 'log' do. '^.(%A)'
 case. 'tanh' do. '7&o.(%A)'
 case. 'sum' do. '+/(%A)'
 case. 'max' do. '>/(%A)'
 case. 'mp' do. '(%A) mp (%B)'
 case. 'badd' do. '(%A) +"(1) (%B)'
 case. 'tr' do. '|:(%A)'
 case. 'emax' do. '(%A) >. (%B)'
 case. 'rsub1' do. '(%A) -"(1 0) (%B)'
 case. 'rsum1' do. '+/"1 (%A)'
 case. 'rdiv1' do. '(%A) %"(1 0) (%B)'
 case. 'rmax1' do. '>./"1 (%A)'
 case. 'reshape' do. '(%B) $ ,(%A)'
 case. 'take' do. '(%B) {."1 (%A)'
 case. 'drop' do. '(%B) }."1 (%A)'
 case. 'gather' do. '(,%B) { (%A)'
 case. 'rsumr' do. '((+/) " 1) (((i. # $ (%A)) -. (%B)) , (%B)) |: (%A)'
 case. do. ('UNKNOWN_' , y)
 end.
)

NB. boolean mask of loss-y ancestors (skips dead subgraphs)
adcreach=: 3 : 0
 m=. ADN $ 0
 m=. 1 y} m
 chg=. 1
 while. chg do.
  chg=. 0
  for_i. I. m do.
   if. -. (nop i) -: 'leaf' do.
    for_j. npar i do.
     if. (j < ADN) *. -. j { m do.
      m=. 1 j} m
      chg=. 1
     end.
    end.
   end.
  end.
 end.
 m
)

NB. x = loss node id, y = integer suffix; defines adcF<y> / adcB<y>
ADGEN=: 4 : 0
 sfx=. ": y
 fn=. 'adcF' , sfx
 bn=. 'adcB' , sfx
 L=. #adleafids
 lossid=. x
 m=. adcreach lossid
 NB. ---------- bake constants (compile time) ----------
 for_i. i. ADN do.
  if. ((nop i) -: 'leaf') *. i >: L do.
   ('adcc' , (":i)) =: nval i
  end.
 end.

 NB. ---------- forward lines ----------
 fln=. ''
 for_i. i. ADN do.
  o=. nop i
  if. o -: 'leaf' do.
   if. i < L do.
    nm=. > 0 { > i { adleafids
    fln=. fln , ('v' , (":i) , '=. ' , nm) , LF
   else.
    fln=. fln , ('v' , (":i) , '=. adcc' , (":i)) , LF
   end.
  else.
   'a b'=. npar i
   expr=. (adcfdesc o) adcsubst (('v' , ":a) ; ('v' , ":b))
   fln=. fln , ('v' , (":i) , '=. ' , expr) , LF
  end.
 end.

 NB. ---------- backward lines (forward recompute + VJP walk) ----------
 bln=. fln
 for_i. i. ADN do.
  if. (i < L) do.
   bln=. bln , ('g' , (":i) , '=. 0') , LF
  else.
   if. (i { m) *. -. (nop i) -: 'leaf' do.
    bln=. bln , ('g' , (":i) , '=. 0') , LF
   end.
  end.
 end.
 bln=. bln , ('g' , (":lossid) , '=. 1') , LF

 i=. <: ADN
 while. i >: 0 do.
  if. (i { m) *. -. (nop i) -: 'leaf' do.
   o=. nop i
   'a b'=. npar i
   gi=. 'g' , ":i
   va=. 'v' , ":a
   vb=. 'v' , ":b
   ga=. 'g' , ":a
   gb=. 'g' , ":b
   NB. constant-leaf parents get no gradient
   ac=. ((nop a) -: 'leaf') *. a >: L
   bc=. ((nop b) -: 'leaf') *. b >: L
   lin=. ''
   select. o
   case. 'add' do.
    if. -. ac do. lin=. lin , <(ga , '=. ' , ga , ' + (' , va , ' adcshrink ' , gi , ')') end.
    if. -. bc do. lin=. lin , <(gb , '=. ' , gb , ' + (' , vb , ' adcshrink ' , gi , ')') end.
   case. 'sub' do.
    if. -. ac do. lin=. lin , <(ga , '=. ' , ga , ' + (' , va , ' adcshrink ' , gi , ')') end.
    if. -. bc do. lin=. lin , <(gb , '=. ' , gb , ' - (' , vb , ' adcshrink ' , gi , ')') end.
   case. 'neg' do.
    if. -. ac do. lin=. lin , <(ga , '=. ' , ga , ' - ' , gi) end.
   case. 'mul' do.
    if. -. ac do. lin=. lin , <(ga , '=. ' , ga , ' + (' , va , ' adcshrink (' , gi , ' * ' , vb , '))') end.
    if. -. bc do. lin=. lin , <(gb , '=. ' , gb , ' + (' , vb , ' adcshrink (' , gi , ' * ' , va , '))') end.
   case. 'div' do.
    if. -. ac do. lin=. lin , <(ga , '=. ' , ga , ' + (' , va , ' adcshrink (' , gi , ' % ' , vb , '))') end.
    if. -. bc do. lin=. lin , <(gb , '=. ' , gb , ' + (' , vb , ' adcshrink ((-' , gi , ') * (' , va , ' % (' , vb , ') * (' , vb , '))))') end.
   case. 'sq' do.
    if. -. ac do. lin=. lin , <(ga , '=. ' , ga , ' + (' , gi , ' * 2 * ' , va , ')') end.
   case. 'pow' do.
    if. -. ac do. lin=. lin , <(ga , '=. ' , ga , ' + (' , gi , ' * ' , vb , ' * ' , va , ' ^ (' , vb , ') - 1)') end.
   case. 'exp' do.
    if. -. ac do. lin=. lin , <(ga , '=. ' , ga , ' + (' , gi , ' * ^ ' , va , ')') end.
   case. 'log' do.
    if. -. ac do. lin=. lin , <(ga , '=. ' , ga , ' + (' , gi , ' % ' , va , ')') end.
   case. 'tanh' do.
    if. -. ac do. lin=. lin , <(ga , '=. ' , ga , ' + (' , gi , ' * (1 - *: 7&o. ' , va , '))') end.
   case. 'sum' do.
    if. -. ac do. lin=. lin , <(ga , '=. ' , ga , ' + (($' , va , ') $ ' , gi , ')') end.
   case. 'max' do.
    NB. mirrors ad.ijs nback 'max' (one-hot mask, incoming g dropped)
    if. -. ac do. lin=. lin , <(ga , '=. ' , ga , ' + ((' , va , ') = >./ ' , va , ')') end.
   case. 'badd' do.
    if. -. ac do. lin=. lin , <(ga , '=. ' , ga , ' + ' , gi) end.
    if. -. bc do. lin=. lin , <(gb , '=. ' , gb , ' + +/ ' , gi) end.
   case. 'mp' do.
    if. -. ac do. lin=. lin , <(ga , '=. ' , ga , ' + (' , gi , ' mp |: ' , vb , ')') end.
    if. -. bc do. lin=. lin , <(gb , '=. ' , gb , ' + ((|: ' , va , ') mp ' , gi , ')') end.
   case. 'tr' do.
    if. -. ac do. lin=. lin , <(ga , '=. ' , ga , ' + |: ' , gi) end.
   case. 'emax' do.
    if. -. ac do. lin=. lin , <(ga , '=. ' , ga , ' + (' , gi , ' * (' , va , ' >: ' , vb , '))') end.
    if. -. bc do. lin=. lin , <(gb , '=. ' , gb , ' + (' , gi , ' * (' , vb , ' >: ' , va , '))') end.
   case. 'rsub1' do.
    if. -. ac do. lin=. lin , <(ga , '=. ' , ga , ' + ' , gi) end.
    if. -. bc do. lin=. lin , <(gb , '=. ' , gb , ' + (- +/"1 ' , gi , ')') end.
   case. 'rsum1' do.
    if. -. ac do.
     lin=. lin , <('pa2=. $' , va)
     lin=. lin , <('eg=. ({: pa2) # ' , gi)
     lin=. lin , <(ga , '=. ' , ga , ' + (pa2 $ eg)')
    end.
   case. 'rdiv1' do.
    if. -. ac do. lin=. lin , <(ga , '=. ' , ga , ' + (' , gi , ' %"(1 0) ' , vb , ')') end.
    if. -. bc do. lin=. lin , <(gb , '=. ' , gb , ' + (- +/"1 (' , gi , ' * (' , va , ' %"(1 0) ((' , vb , ') * (' , vb , ')))))') end.
   case. 'rmax1' do.
    if. -. ac do. lin=. lin , <(ga , '=. ' , ga , ' + (' , gi , ' * ((' , va , ') ="(1 0) ' , 'v' , (":i) , '))') end.
   case. 'reshape' do.
    if. -. ac do. lin=. lin , <(ga , '=. ' , ga , ' + (($' , va , ') $ , ' , gi , ')') end.
   case. 'take' do.
    if. -. ac do.
     lin=. lin , <('tzn=. ' , vb)
     lin=. lin , <('tzs=. $' , va)
     lin=. lin , <('tzz=. ((0 { tzs) , tzn) $ 0.0')
     lin=. lin , <(ga , '=. ' , ga , ' + (' , gi , ' ,"1 tzz)')
    end.
   case. 'drop' do.
    if. -. ac do.
     lin=. lin , <('pzn=. ' , vb)
     lin=. lin , <('pzs=. $' , va)
     lin=. lin , <('pzz=. ((0 { pzs) , pzn) $ 0.0')
     lin=. lin , <(ga , '=. ' , ga , ' + (pzz ,"1 ' , gi , ')')
    end.
   case. 'gather' do.
    if. -. ac do.
     NB. scatter-accumulate: (i.nrows)="(0 1) idx matrix mp g (must SUM dups)
     lin=. lin , <('grn=. 0 { $' , va)
     lin=. lin , <(ga , '=. ' , ga , ' + (((i. grn) ="(0 1) (,' , vb , ')) mp ' , gi , ')')
    end.
   case. 'rsumr' do.
    if. -. ac do.
     NB. mirror interpreted VJP: rotate axis k to the end, replicate each
     NB. frame scalar across the reduced axis, reshape, un-rotate.
     lin=. lin , <('rsash=. $' , va)
     lin=. lin , <('rsk=. ' , vb)
     lin=. lin , <('rspv=. ((i. # rsash) -. rsk) , rsk')
     lin=. lin , <('rsrsh=. rsash { ~ rspv')
     lin=. lin , <('rseg=. ({: rsrsh) # , ' , gi)
     lin=. lin , <(ga , '=. ' , ga , ' + ((/: rspv) |: (rsrsh $ rseg))')
    end.
   end.
   for_l. lin do. bln=. bln , (>l) , LF end.
  end.
  i=. <:i
 end.

 NB. ---------- exports ----------
 bln=. bln , ('adcgL=: v' , (":lossid)) , LF
 for_k. i. L do.
  bln=. bln , ('adcg' , (":k) , '=: g' , (":k)) , LF
 end.
 bln=. bln , 'adcgL' , LF
 (0!:10) (bn , '=: 3 : 0' , LF , bln , ')')

 fln=. fln , ('v' , (":lossid)) , LF
 (0!:10) (fn , '=: 3 : 0' , LF , fln , ')')
 1
)