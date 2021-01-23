; Hinode files from
; https://csac.hao.ucar.edu/sp_data.php?date1=2010-01-01&date2=2010-01-31

files=file_search('profiles/*')
nx=n_elements(files)
ii=readfits(files[0],header)
nlam=(size(ii))[1]
ny=(size(ii))[2]
cont=fltarr(nx,ny)
stki=fltarr(nx,ny,nlam)
stkq=fltarr(nx,ny,nlam)
stku=fltarr(nx,ny,nlam)
stkv=fltarr(nx,ny,nlam)
for ix=0,nx-1 do begin
   ii=readfits(files[ix],header, /silent)
   ii=float(ii)
   i=ii[*,*,0]
   if (min(i) lt 0) then $
      i(where(i lt 0))==i(where(i lt 0))+65536.
   
   cont[ix,*]=total(i[50:55,*],1)/6.
   if ix/10 eq ix/10. then print,'Reading pos:',ix
   stki[ix,*,*]=transpose(i)
   stkq[ix,*,*]=transpose(ii[*,*,1])
   stku[ix,*,*]=transpose(ii[*,*,2])
   stkv[ix,*,*]=transpose(ii[*,*,3])
endfor

qscont=median(cont[0:300,*])
stki=stki/qscont & stkq=stkq/qscont & stku=stku/qscont & stkv=stkv/qscont & 

lambda=6300.89+.0214*findgen(112)


save,file='obs.prof.idl',stki,stkq,stku,stkv

end
