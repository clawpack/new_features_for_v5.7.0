
subroutine fgmax_interpolate(mx,my,meqn,mbc,maux,q,aux,dx,dy, &
           xlower,ylower,ifg,level,fg_values,i1,i2,j1,j2, &
           fg_klist,fg_klist_length,fg_npts)

    ! Given a grid patch, return fg_values containing values interpolated
    ! to the fixed grid fg => FG_fgrids(ifg).
    ! If there are aux arrays, also set fg%aux at any points where it has
    ! not yet been set on this level.

    ! This version uses piecewise constant interpolation, in other words the
    ! cell value for the cell containing the fixed grid point is returned,
    ! without using values in neighboring cells.

    use fgmax_module

    implicit none
    integer, intent(in) :: mx,my,meqn,mbc,maux,ifg,level,fg_npts
    integer, intent(in) :: fg_klist(fg_npts), fg_klist_length
    real(kind=8), intent(in) :: q(meqn, 1-mbc:mx+mbc, 1-mbc:my+mbc)
    real(kind=8), intent(in) :: aux(maux, 1-mbc:mx+mbc, 1-mbc:my+mbc)
    real(kind=8), intent(in) :: dx,dy,xlower,ylower
    !real(kind=8), dimension(:,:), allocatable, intent(inout) :: fg_values
    !logical, dimension(:), allocatable, intent(inout) :: mask_fgrid
    real(kind=8), intent(inout) :: fg_values(FG_NUM_VAL,fg_klist_length)
    integer, intent(in) :: i1,i2,j1,j2

    type(fgrid), pointer :: fg
    integer :: i,j,k,mv,ma
    integer :: indexk
    real(kind=8), allocatable, dimension(:,:,:) :: values
    real(kind=8) :: x,y
    logical :: debug

    debug = FG_DEBUG
    if (debug) then
        write(61,*) '========================================'
        write(61,*) 'In fgmax_interpolate, Level = ',level
        endif

    fg => FG_fgrids(ifg)
    !fg_npts = fg%npts  ! now an input parameter
    !write(61,*) '++++ interpolate xNbb', fg%x1bb, fg%x2bb
    
    !write(6,61) fg%fgno, fg%npts
 61 format('Updating fgrid number ',i2,' with',i7,' points')

    ! array values with same size as patch:
    allocate(values(FG_NUM_VAL, 1-mbc:mx+mbc, 1-mbc:my+mbc))


    if (mbc < 1) then
        write(6,*) '*** mbc >= 1 required by fgmax_interpolate'
        stop
        endif


    ! Set the array values to be the values that we want to update:
    ! Note that values will be set properly only where i,j in [i1,i2,j1,j2]
    values = FG_NOTSET
    call fgmax_values(mx,my,meqn,mbc,maux,q,aux,dx,dy, &
                   xlower,ylower,i1,i2,j1,j2,values)
    if (debug) then
        write(65,*) '+++ i,j,x,y,values(1,i,j) '
        write(65,*) '    at points where i,j in [i1,i2,j1,j2]'
        do i=i1,i2
            x = xlower + (i-0.5d0)*dx
            do j=j1,j2
                y = ylower + (j-0.5d0)*dy
                write(65,65) i,j,x,y,values(1,i,j)
 65             format(2i4,2d16.6,d20.9)
                enddo
            enddo
        endif
        

    ! Determine indices of cell containing fixed grid point (x(k),y(k))

    !do i=1,fg_klist_length 
    !    k = fg_klist(i)
    !    ik(k) = int((fg%x(k) - xlower + dx) / dx)
    !    jk(k) = int((fg%y(k) - ylower + dy) / dy)
    !    enddo

    ! fix this loop?
    if (debug) then
        write(61,*) '+++ fg_klist: k,x,y,ik,jk: '
        endif


    ! reordered loops, now outer loop is over fgmax points on this patch
    do indexk=1,fg_klist_length 
    
        k = fg_klist(indexk)
        
        ! compute i,j directly without storing in ik,jk, since now only
        ! need to be computed once with new loop order    
        i = int((fg%x(k) - xlower + dx) / dx)
        j = int((fg%y(k) - ylower + dy) / dy)
        
        if (debug) then
            write(61,62) k,fg%x(k),fg%y(k), i,j
62          format(i4,2d16.6,2i6,2d16.6)
            endif

        do mv=1,FG_NUM_VAL
            ! loop over the different values we want to monitor
                
            ! Do not print the warning message below
            ! It is possible that i==0 or mx+1, or j==0 or my+1
            ! if the fg%x(k) or fg%y(k) was approx on the boundary of
            ! the patch, but in this case it should be ok to use the
            ! the ghost cell value. 
            !if ((i==mx+1) .or. (j==my+1)) then
            !    write(6,*) '**** Warning from fgmax_interpolate:'
            !    write(6,*) '**** Expected i <= mx, j<= my'
            !    write(6,*) 'i,j,mx,my: ',i,j,mx,my
            !    endif

            fg_values(mv,indexk) = values(mv,i,j) 
            enddo

        ! Set the fg%aux(level,:,:) if this hasn't yet been set.
        if ((maux > 0) .and. (.not. fg%auxdone(level))) then
            ! at least some points do not yet have fg%aux set on this level

            do ma=1,FG_NUM_AUX
                if ((fg%aux(level,ma,k) == FG_NOTSET)) then
                    fg%aux(level,ma,k) = aux(ma,i,j)
                    endif
                enddo
            endif

        enddo ! loop on indexk

    ! note computing minval checks all fg%npts points, better way?
    if (minval(fg%aux(level,1,:)) > FG_NOTSET) then
        ! Done with aux arrays at all fgrid points on this level
        !print *, '+++ level,fg%aux:',level,fg%aux(level,1,1)
        fg%auxdone(level) = .true.
        endif
        
    !deallocate(ik,jk,mask_patch)

end subroutine fgmax_interpolate
