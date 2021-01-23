; Makes use of IDL model procedures in NICOLE

;instrument='visp'
instrument='hinode'

stray_frac=0.10 ; fraction of stray light
modelfiles=['hsra.model','mmaltby.model','falc.model','valc.model']
nfiles=n_elements(modelfiles)
nperfile= 1000/nfiles 
nx=nfiles*nperfile
ny=1000

tau=(-findgen(70)+20)/10.
nz=n_elements(tau)

npars=9 ; T_0, T_1, T_2, T_3, T_4, B_0, Bx_0, By_0, v_0
itaus5=[20,30,40,50,60] ; log(tau)=[0,-1,-2,-3,-4]

If instrument eq 'visp' then begin
   fwhm=10.824                ; mA from ViSP configuration 
   sigma=fwhm/2.355
   vmac=sigma*1e-3/6302.*3e8    ; spectral psf in cm/s
   lam=findgen(175)*.010824+6301.1
   spawn,'rm Instrumental_profile.dat'
   spawn,'cp NICOLE.input_visp NICOLE.input'
endif else begin ; hinode
   vmac=0.
   deltalambda=.0214
   lam=findgen(112)*deltalambda+6300.89
   spawn,'cp NICOLE.input_hinode NICOLE.input'
endelse

models=new_model(nx,ny,nz)
params=fltarr(nx,ny,npars)
pertur=fltarr(nz)
ix=0
for ifile=0,n_elements(modelfiles)-1 do begin
   interpolate_model,modelfiles[ifile],'tmp.model',tau
   refmodel=read_model('tmp.model')
   refmodel.v_mac=vmac
   refmodel.v_mic=0.
   for iperfile=0,nperfile-1 do begin
      for iy=0,ny-1 do begin
         model=refmodel
         model.v_mac=(model.v_mac+randomu(kk)*.6e5)>0
         deltatau=max(tau)-min(tau)
         ; Temperature
         ipar=0
         deltas=[randomn(kk),randomn(kk),randomn(kk),randomn(kk)]*1500.
         deltas=smooth(deltas,2)
         deltas_x=findgen(n_elements(deltas))*deltatau/n_elements(deltas)+min(tau)
         pertur=interpol(deltas,deltas_x,tau)
         if (ix eq 0 and iy eq 0) then pertur=pertur*0.
         model.t=model.t+pertur
         model.t=model.t>2000   ; safety check
         
         params[ix,iy,ipar:ipar+4]=model.t[[itaus5[0],itaus5[1],itaus5[2],itaus5[3],itaus5[4]]]
         ipar=ipar+5

         ; B_0 
         ff=1.
         if randomu(kk) lt .5 then ff=1.-randomu(kk)*.9
         model.ffactor=ff
         delta=randomu(kk)*500 ; quiet sun
         if randomu(kk) lt .5 then $ ; active
                     delta=randomu(kk)*2e3
         if randomu(kk) lt .1 then $ ; extreme values
            delta=randomu(kk)*6e3
         if randomu(kk) lt .1 then $ ; zero value
            delta=0.
         if randomu(kk) lt .5 then delta=-delta ; flip sign with 50% chance
         delta_grad=0.
         if randomu(kk) lt .5 then delta_grad=randomu(kk)*2e3/4. ; 50% chance of gradient
         if randomu(kk) lt .5 then delta_grad=-delta_grad ; flip sign with 50% chance
         pertur=delta+(tau)/4.*delta_grad
         if (ix eq 0 and iy eq 0) then pertur=pertur*0.
         model.b_los_z=model.b_los_z+pertur
         params[ix,iy,ipar]=pertur[itaus[0]]*ff
         ipar=ipar+1
         ; Bx_0
         delta=randomu(kk)*500 ; quiet sun
         if randomu(kk) lt .5 then $ ; active
                     delta=randomu(kk)*2e3
         if randomu(kk) lt .1 then $ ; extreme values
                     delta=randomu(kk)*4e3
         if randomu(kk) lt .1 then $ ; zero value
            delta=0.
         ;if randomu(kk) lt .5 then delta=-delta ; flip sign with 50% chance
         pertur=delta
         if (ix eq 0 and iy eq 0) then pertur=pertur*0.
         model.b_los_x=model.b_los_x+pertur
         params[ix,iy,ipar]=pertur*ff
         ipar=ipar+1
         ; By_0
         delta=randomu(kk)*500 ; quiet sun
         if randomu(kk) lt .5 then $ ; active
                     delta=randomu(kk)*2e3
         if randomu(kk) lt .1 then $ ; extreme values
                     delta=randomu(kk)*4e3
         if randomu(kk) lt .1 then $ ; zero value
            delta=0.
         if randomu(kk) lt .5 then delta=-delta ; flip sign with 50% chance
         pertur=delta
         if (ix eq 0 and iy eq 0) then pertur=pertur*0.
         model.b_los_y=model.b_los_y+pertur
         params[ix,iy,ipar]=pertur*ff
         ipar=ipar+1
         ; v_0 
         delta=randomu(kk)*4e5
;         if randomu(kk) lt .1 then $ ; extreme values
;                     delta=randomn(kk)*7e3
         if randomu(kk) lt .5 then delta=-delta ; flip sign with 50% chance
         delta_grad=0.
         if randomu(kk) lt .5 then delta_grad=randomu(kk)*2e5/4. ; 50% chance of gradient
         if randomu(kk) lt .5 then delta_grad=-delta_grad ; flip sign with 50% chance
         pertur=delta+(tau)/4.*delta_grad
         if (ix eq 0 and iy eq 0) then pertur=pertur*0.
         model.v_los=model.v_los+pertur
         ;
         model_column,models,ix,iy,model,0,0
         params[ix,iy,ipar]=pertur[itaus[0]]
         ipar=ipar+1
      endfor
      ix=ix+1
   endfor
endfor
print,'Done. Writing files'
idl_to_nicole,file='modelin.model',model=models
params_valid=params[nx-1,*,*]
params=params[0:nx-2,*,*]
save,file='params.idl_'+instrument,params,/compress
params=params_valid
save,file='../validation/params.idl_'+instrument,params,/compress

spawn,'python2 ./run_nicole.py'

m=read_model('modelin.model')
i=read_profile('modelout.prof',q,u,v)
strayprof=reform(i[0,0,*])

if instrument eq 'visp' then begin

   l11=6301.8
   l12=6302.1
   l21=6302.65
   l22=6302.9

   i11=(whereq(lam,l11))[0]
   i12=(whereq(lam,l12))[0]
   ni1=float(i12-i11+1)
   i21=(whereq(lam,l21))[0]
   i22=(whereq(lam,l22))[0]
   ni2=float(i22-i21+1)

   nx=(size(i))(1)
   ny=(size(i))(2)

   for ix=0,nx-1 do for iy=0,ny-1 do i[ix,iy,i11:i12]=i[ix,iy,i11]+((i[ix,iy,i12]-i[ix,iy,i11])/ni1)[0]*findgen(ni1)
   for ix=0,nx-1 do for iy=0,ny-1 do i[ix,iy,i21:i22]=i[ix,iy,i21]+((i[ix,iy,i22]-i[ix,iy,i21])/ni2)[0]*findgen(ni2)
endif 

for ilam=0,n_elements(lam)-1 do i[*,*,ilam]=strayprof[ilam]*stray_frac+i[*,*,ilam]*(1.-stray_frac)
for ilam=0,n_elements(lam)-1 do q[*,*,ilam]=q[*,*,ilam]*m.ffactor[*,*]*(1.-stray_frac)
for ilam=0,n_elements(lam)-1 do u[*,*,ilam]=u[*,*,ilam]*m.ffactor[*,*]*(1.-stray_frac)
for ilam=0,n_elements(lam)-1 do v[*,*,ilam]=v[*,*,ilam]*m.ffactor[*,*]*(1.-stray_frac)

stki=i[nx-1,*,*]
stkq=q[nx-1,*,*]
stku=u[nx-1,*,*]
stkv=v[nx-1,*,*]
save,file='../validation/obs.prof_'+instrument+'.idl',stki,stkq,stku,stkv,/compress


stki=i[0:nx-2,*,*]
stkq=q[0:nx-2,*,*]
stku=u[0:nx-2,*,*]
stkv=v[0:nx-2,*,*]
save,file='database.prof_'+instrument+'.idl',stki,stkq,stku,stkv,/compress

print,'Done. Dont forget to copy database.prof_'+instrument+'.idl and params.idl_'+instrument+' to database.prof.idl and params.idl'



end
