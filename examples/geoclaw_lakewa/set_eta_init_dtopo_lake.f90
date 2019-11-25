
subroutine set_eta_init(mbc,mx,my,xlow,ylow,dx,dy,t,veta)

    ! set variable eta_int in veta, 
    ! called by qinit and also by filpatch and filval when refining.

    ! set surface elevation eta differently in Lake Washington than
    ! in Puget Sound, and also adjust by dtopo co-seismic displacement

    use topo_module
    use geoclaw_module, only: sea_level

    implicit none

    ! Arguments
    integer, intent(in) :: mbc,mx,my
    real(kind=8), intent(in) :: xlow,ylow,dx,dy,t
    real(kind=8), intent(inout) :: veta(1-mbc:mx+mbc,1-mbc:my+mbc)

    ! parameters defining boundary between Sound and Lake:
    integer, parameter :: mbdry = 7
    real(kind=8), parameter,dimension(mbdry) :: &
        xbdry = (/-122.2, -122.2, -122.3, -122.31, -122.3974, &
                  -122.3974, -122.28 /) 
    real(kind=8), parameter,dimension(mbdry) :: &
        ybdry = (/47.39, 47.46,  47.54, 47.60,  47.65, &
                  47.68, 47.82 /)
    
    ! Local
    integer :: i,j,i1,i2,j1,j2,idtopo,jdtopo,kdtopo, &
               index0_dtopowork,ij,m,k
    real(kind=8) :: x,y,xofy

    veta = sea_level  ! initialize to sea_level, update below at some (i,j)

    ! Assume sea_level set above is level of Lake Washington
    ! If not in lake, reset to 0 (= MHW in Sound):

    ! xbdry,ybdry defines a piecewise linear N-S running curve,
    ! where the lake is only to the east of this curve, Sound to west

    do i=1,mx
        x = xlow + (i-0.5d0)*dx
        do j=1,my
            y = ylow + (j-0.5d0)*dy
            do k=1,mbdry
                if ((y >= ybdry(k)) .and. (y < ybdry(k+1))) then
                    xofy = xbdry(k) + (xbdry(k+1)-xbdry(k)) * &
                            (y-ybdry(k))/((ybdry(k+1)-ybdry(k)))
                    if (x < xofy) then
                        veta(i,j) = 0.d0  ! sea level of Sound
                        endif
                    exit
                    endif
                enddo
            enddo
        enddo
                    

    ! now adjust for dtopo:

    if (num_dtopo == 0) return

    do m=1,num_dtopo
        if (t < t0dtopo(m)) cycle  ! this dtopo isn't moving yet
        
        ! compute indices in patch that overlap with this dtopo,
        ! cycling to the next dtopo array if we discover there's no overlap:
        
        i1 = max(int(floor((xlowdtopo(m) - xlow)/dx)), 1)
        if (i1 >= mx) cycle
        
        i2 = min(int(floor((xhidtopo(m) - xlow)/dx)), mx)
        if (i2 < 1) cycle

        j1 = max(int(floor((ylowdtopo(m) - ylow)/dy)), 1)
        if (j1 >= my) cycle
        
        j2 = min(int(floor((yhidtopo(m) - ylow)/dy)), my)
        if (j2 < 1) cycle
    
        ! There is some overlap of dtopo with this patch
        ! Next figure out index into time-dependent dtopo based on t:

        if (mtdtopo(m) == 1) then
            ! Special case: instantaneous displacement at one instant in time
            kdtopo = 1
          else
            kdtopo = int(floor((t-t0dtopo(m))/dtdtopo(m)))+1
            kdtopo = min(kdtopo,mtdtopo(m))
            kdtopo = max(kdtopo,1)
          endif

        index0_dtopowork = i0dtopo(m) + (kdtopo-1)*mxdtopo(m)*mydtopo(m)
        !write(6,*) '+++ index0_dtopowork = ',index0_dtopowork

        ! Adjust eta_init by dtopo on part of patch that overlaps dtopo.
        ! Code below assumes dtopo is smooth enough that we can just 
        ! evaluate at one point in space and time, not doing interpolation.
        ! This could be improved as in topo_update.f, but probably not
        ! necessary (??).

        do i=i1,i2
            x = xlow + (i-0.5d0)*dx
            do j=j1,j2
                y = ylow + (j-0.5d0)*dy
                idtopo = int(floor((x-xlowdtopo(m))/dxdtopo(m))) + 1
                idtopo = max(1, min(mxdtopo(m)-1, idtopo))
                jdtopo = int(floor((yhidtopo(m)-y)/dydtopo(m))) + 1
                jdtopo = max(1, min(mydtopo(m)-1, jdtopo))
                ij = index0_dtopowork + (jdtopo-1)*mxdtopo(m) + idtopo-1
                veta(i,j) = veta(i,j) + dtopowork(ij)

                ! print subsidence at location of interest for debugging:
                if (.false. .and.(x>0.0).and.(x<0.0006).and.(y>0.0) &
                     .and. (y<0.0006)) then
                       write(6,*) '+++ i,j,veta:', i,j,veta(i,j)
                       !write(6,*) '+++ idtopo,jdtopo,ij: ', idtopo,jdtopo,ij
                      endif
                enddo
            enddo

        enddo
    return

end subroutine set_eta_init
