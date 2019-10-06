
subroutine qinit(meqn,mbc,mx,my,xlower,ylower,dx,dy,q,maux,aux)
    
    use qinit_module, only: qinit_type,add_perturbation
    use qinit_module, only: wet_mask,use_wet_mask,mx_wet, my_wet
    use qinit_module, only: xlow_wet, ylow_wet, xhi_wet, yhi_wet, dx_wet, dy_wet
    use geoclaw_module, only: sea_level
    use amr_module, only: t0
    use topo_module, only: variable_eta_init
    
    implicit none
    
    ! Subroutine arguments
    integer, intent(in) :: meqn,mbc,mx,my,maux
    real(kind=8), intent(in) :: xlower,ylower,dx,dy
    real(kind=8), intent(inout) :: q(meqn,1-mbc:mx+mbc,1-mbc:my+mbc)
    real(kind=8), intent(inout) :: aux(maux,1-mbc:mx+mbc,1-mbc:my+mbc)
    
    ! Locals
    integer :: i,j,m, ii,jj
    real(kind=8) :: x,y
    real(kind=8) :: veta(1-mbc:mx+mbc,1-mbc:my+mbc)
    real(kind=8) :: ddxy
    
    
    if (variable_eta_init) then
        ! Set initial surface eta based on eta_init
        call set_eta_init(mbc,mx,my,xlower,ylower,dx,dy,t0,veta)
      else
        veta = sea_level  ! same value everywhere
      endif

    q(2:3,:,:) = 0.d0   ! set momenta to zero

    forall(i=1:mx, j=1:my)
        q(1,i,j) = max(0.d0, veta(i,j) - aux(1,i,j))
    end forall

    if (use_wet_mask) then
     ! only use the wet_mask if it specified on a grid that matches the 
     ! resolution of this patch, since we only check the cell center:
     ddxy = max(abs(dx-dx_wet), abs(dy-dy_wet))
     if (ddxy < 0.01d0*min(dx_wet,dy_wet)) then
       do i=1,mx
          x = xlower + (i-0.5d0)*dx
          ii = int((x - xlow_wet + 1d-7) / dx_wet)
          do j=1,my
              y = ylower + (j-0.5d0)*dy
              jj = int((y - ylow_wet + 1d-7) / dy_wet)
              jj = my_wet - jj  ! since index 1 corresponds to north edge
              if ((ii>=1) .and. (ii<=mx_wet) .and. &
                  (jj>=1) .and. (jj<=my_wet)) then
                  ! grid cell lies in region covered by wet_mask,
                  ! check if this cell is forced to be dry 
                  ! Otherwise don't change value set above:                  
                  if (wet_mask(ii,jj) == 0) then
                      q(1,i,j) = 0.d0
                      endif
                  endif
          enddo ! loop on j
       enddo ! loop on i
       endif ! dx and dy agree with dx_wet, dy_wet
    endif ! use_wet_mask

    if (dx <= 1.d0/3600.d0) then
     do i=1,mx
       x = xlower + (i-0.5d0)*dx
       do j=1,my
           y = ylower + (j-0.5d0)*dy
           ! special case of eastern Skagit delta:
           if (x>-122.33 .and. y>48.25 .and. y<48.4) then
              q(1,i,j) = 0.d0
              endif
           ! special case SE of Samish Island:
           if (x>-122.5 .and. y>48.52 .and. y<48.57) then
              q(1,i,j) = 0.d0
              endif

         enddo ! loop on j
      enddo ! loop on i
    endif
    
    ! Add perturbation to initial conditions
    if (qinit_type > 0) then
        call add_perturbation(meqn,mbc,mx,my,xlower,ylower,dx,dy,q,maux,aux)
    endif

    if (.false.) then
        open(23, file='fort.aux',status='unknown',form='formatted')
        print *,'Writing out aux arrays'
        print *,' '
        do j=1,my
            do i=1,mx
                write(23,*) i,j,(q(m,i,j),m=1,meqn)
            enddo
        enddo
        close(23)
    endif
    
end subroutine qinit