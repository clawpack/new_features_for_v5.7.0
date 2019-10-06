module qinit_module

    use amr_module, only: rinfinity

    implicit none
    save

    logical :: module_setup = .false.
    
    ! Type of q initialization
    integer, public :: qinit_type
    
    ! Work array
    real(kind=8), private, allocatable :: qinit(:)

    ! Geometry
    real(kind=8) :: x_low_qinit
    real(kind=8) :: y_low_qinit
    real(kind=8) :: t_low_qinit
    real(kind=8) :: x_hi_qinit
    real(kind=8) :: y_hi_qinit
    real(kind=8) :: t_hi_qinit
    real(kind=8) :: dx_qinit
    real(kind=8) :: dy_qinit
    
    integer, private :: mx_qinit
    integer, private :: my_qinit
    integer :: min_level_qinit
    integer :: max_level_qinit

    ! for initializing using wet_mask to indicate wet regions:

    integer :: mx_wet, my_wet
    real(kind=8) :: xlow_wet, ylow_wet, xhi_wet, yhi_wet, dx_wet, dy_wet
    integer(kind=1), allocatable :: wet_mask(:,:)
    logical :: use_wet_mask
    real(kind=8) :: t_stays_dry  ! always use mask up to this time

    ! to initialize using different initial eta values in different regions:
    integer :: etain_mx, etain_my
    real(kind=8) :: etain_dx, etain_dy
    real(kind=8), allocatable :: etain_x(:), etain_y(:), etain_eta(:,:)


contains

    subroutine set_qinit(fname)
    
        use geoclaw_module, only: GEO_PARM_UNIT
    
        implicit none
        
        ! Subroutine arguments
        character(len=*), optional, intent(in) :: fname
        
        ! File handling
        integer, parameter :: unit = 7
        character(len=150) :: qinit_fname
        
        if (.not.module_setup) then
            write(GEO_PARM_UNIT,*) ' '
            write(GEO_PARM_UNIT,*) '--------------------------------------------'
            write(GEO_PARM_UNIT,*) 'SETQINIT:'
            write(GEO_PARM_UNIT,*) '-------------'
            
            ! Open the data file
            if (present(fname)) then
                call opendatafile(unit,fname)
            else
                call opendatafile(unit,"qinit.data")
            endif
            
            read(unit,"(i1)") qinit_type
            if (qinit_type == 0) then
                ! No perturbation specified
                write(GEO_PARM_UNIT,*)  '  qinit_type = 0, no perturbation'
                print *,'  qinit_type = 0, no perturbation'
                return
            endif
            read(unit,*) qinit_fname
            read(unit,"(2i2)") min_level_qinit, max_level_qinit

            write(GEO_PARM_UNIT,*) '   min_level, max_level, qinit_fname:'
            write(GEO_PARM_UNIT,*)  min_level_qinit, max_level_qinit, qinit_fname
            
            call read_qinit(qinit_fname)

            module_setup = .true.
        end if
    
    end subroutine set_qinit


    subroutine add_perturbation(meqn,mbc,mx,my,xlow_patch,ylow_patch,dx,dy,q,maux,aux)
    
        use geoclaw_module, only: sea_level, coordinate_system
        use amr_module, only: mcapa
    
        implicit none
    
        ! Subroutine arguments
        integer, intent(in) :: meqn,mbc,mx,my,maux
        real(kind=8), intent(in) :: xlow_patch,ylow_patch,dx,dy
        real(kind=8), intent(inout) :: q(meqn,1-mbc:mx+mbc,1-mbc:my+mbc)
        real(kind=8), intent(inout) :: aux(maux,1-mbc:mx+mbc,1-mbc:my+mbc)
        
        ! Local
        integer :: i,j
        real(kind=8) :: ximc,xim,x,xip,xipc,yjmc,yjm,y,yjp,yjpc,dq
        
        ! Topography integral function
        real(kind=8) :: topointegral
        
        if (qinit_type > 0) then
            do i=1-mbc,mx+mbc
                x = xlow_patch + (i-0.5d0)*dx
                xim = x - 0.5d0*dx
                xip = x + 0.5d0*dx
                do j=1-mbc,my+mbc
                    y = ylow_patch + (j-0.5d0)*dy
                    yjm = y - 0.5d0*dy
                    yjp = y + 0.5d0*dy

                    ! Check to see if we are in the qinit region at this grid point
                    if ((xip > x_low_qinit).and.(xim < x_hi_qinit).and.  &
                        (yjp > y_low_qinit).and.(yjm < y_hi_qinit)) then

                        xipc=min(xip,x_hi_qinit)
                        ximc=max(xim,x_low_qinit)

                        yjpc=min(yjp,y_hi_qinit)
                        yjmc=max(yjm,y_low_qinit)

                        dq = topointegral(ximc,xipc,yjmc,yjpc,x_low_qinit, &
                                          y_low_qinit,dx_qinit,dy_qinit,mx_qinit, &
                                          my_qinit,qinit,1)
                        if (coordinate_system == 2) then
                            dq = dq / ((xipc-ximc)*(yjpc-yjmc)*aux(mcapa,i,j))
                        else
                            dq = dq / ((xipc-ximc)*(yjpc-yjmc))
                        endif 

                        if (qinit_type < 4) then 
                            if (aux(1,i,j) <= sea_level) then
                                q(qinit_type,i,j) = q(qinit_type,i,j) + dq
                            endif
                        else if (qinit_type == 4) then
                            q(1,i,j) = max(dq-aux(1,i,j),0.d0)
                        endif
                    endif
                enddo
            enddo
        endif
        
    end subroutine add_perturbation

        
    ! currently only supports one file type:
    ! x,y,z values, one per line in standard order from NW corner to SE
    ! z is perturbation from standard depth h,hu,hv set in qinit_geo,
    ! if iqinit = 1,2, or 3 respectively.
    ! if iqinit = 4, the z column corresponds to the definition of the 
    ! surface elevation eta. The depth is then set as q(i,j,1)=max(eta-b,0)
    subroutine read_qinit(fname)
    
        use geoclaw_module, only: GEO_PARM_UNIT
        
        implicit none
        
        ! Subroutine arguments
        character(len=150) :: fname
        
        ! Data file opening
        integer, parameter :: unit = 19
        integer :: i,num_points,status
        double precision :: x,y
        
        print *,'  '
        print *,'Reading qinit data from file  ', fname
        print *,'  '

        write(GEO_PARM_UNIT,*) '  '
        write(GEO_PARM_UNIT,*) 'Reading qinit data from'
        write(GEO_PARM_UNIT,*) fname
        write(GEO_PARM_UNIT,*) '  '
        
        open(unit=unit, file=fname, iostat=status, status="unknown", &
             form='formatted',action="read")
        if ( status /= 0 ) then
            print *,"Error opening file", fname
            stop
        endif
        
        ! Initialize counters
        num_points = 0
        mx_qinit = 0
        
        ! Read in first values, determines x_low and y_hi
        read(unit,*) x_low_qinit,y_hi_qinit
        num_points = num_points + 1
        mx_qinit = mx_qinit + 1
        
        ! Sweep through first row figuring out mx
        y = y_hi_qinit
        do while (y_hi_qinit == y)
            read(unit,*) x,y
            num_points = num_points + 1
            mx_qinit = mx_qinit + 1
        enddo
        ! We over count by one in the above loop
        mx_qinit = mx_qinit - 1
        
        ! Continue to count the rest of the lines
        do
            read(unit,*,iostat=status) x,y
            if (status /= 0) exit
            num_points = num_points + 1
        enddo
        if (status > 0) then
            print *,"ERROR:  Error reading qinit file ",fname
            stop
        endif
        
        ! Extract rest of geometry
        x_hi_qinit = x
        y_low_qinit = y
        my_qinit = num_points / mx_qinit
        dx_qinit = (x_hi_qinit - x_low_qinit) / (mx_qinit-1)
        dy_qinit = (y_hi_qinit - y_low_qinit) / (my_qinit-1)
        
        rewind(unit)
        allocate(qinit(num_points))
        
        ! Read and store the data this time
        do i=1,num_points
            read(unit,*) x,y,qinit(i)
        enddo
        close(unit)
        
    end subroutine read_qinit

    subroutine read_wet_mask(fname)

        use utility_module, only: parse_values
        character(len=*), intent(in) :: fname
        integer :: iunit,i,j,n
        real(kind=8) :: values(10), nodata_value
        character(len=80) :: str

        iunit = 8
    
        open(unit=iunit,file=fname,status='old',form='formatted')
        !read(iunit,*) t_stays_dry
        !write(6,*) 't_stays_dry = ',t_stays_dry
        read(iunit,*) mx_wet
        read(iunit,*) my_wet
        read(iunit,*) xlow_wet
        read(iunit,*) ylow_wet

        read(iunit,'(a)') str
        call parse_values(str, n, values)
        dx_wet = values(1)
        if (n == 2) then
            dy_wet = values(2)
          else
            dy_wet = dx_wet
          endif

        read(iunit,*) nodata_value
        allocate(wet_mask(mx_wet,my_wet))

        xhi_wet = xlow_wet + mx_wet*dx_wet
        yhi_wet = ylow_wet + my_wet*dy_wet
        write(6,*) '+++ xlow_wet, xhi_wet: ',xlow_wet, xhi_wet
        write(6,*) '+++ ylow_wet, yhi_wet: ',ylow_wet, yhi_wet

        do j=1,my_wet
            read(iunit, *) (wet_mask(i,j), i=1,mx_wet)
            enddo
    
        close(iunit)
        return
    end subroutine read_wet_mask

    
    subroutine read_eta_init(file_name)
        ! To read in file specifying different eta value in at different
        ! locations, then used in qinit function.
        ! Uses etain module variables.
        
        implicit none

        ! Input arguments
        character(len=*), intent(in), optional :: file_name
        
        ! local 
        integer, parameter :: iunit = 7
        integer :: i,j
        real(kind=8) :: nodata_value, xllower, yllower

        if (present(file_name)) then
            open(unit=iunit, file=file_name, status='unknown',&
                      form='formatted')
        else
            open(unit=iunit, file='eta_init.data', status='unknown',&
                      form='formatted')
        endif
        
        read(iunit,*) etain_mx
        !write(6,*) '+++ etain_mx = ',etain_mx
        read(iunit,*) etain_my
        !write(6,*) '+++ etain_my = ',etain_my
        read(iunit,*) xllower
        read(iunit,*) yllower
        read(iunit,*) etain_dx
        etain_dy = etain_dx
        !read(iunit,*) etain_dy
        read(iunit,*) nodata_value
        
        allocate(etain_x(etain_mx), etain_y(etain_my))
        allocate(etain_eta(etain_mx, etain_my))
        
        do i=1,etain_mx
            etain_x(i) = xllower + etain_dx*(i-1)
            enddo
            
        do j=1,etain_my
            etain_y(j) = yllower + etain_dy*(etain_my-j+1)
            read(iunit,*) (etain_eta(i,j),i=1,etain_mx)
            enddo

        
        close(unit=iunit)
    end subroutine read_eta_init

end module qinit_module
