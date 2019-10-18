subroutine setprob()

    use topo_module, only: variable_eta_init
    use qinit_module, only: read_force_dry, use_force_dry, tend_force_dry
    implicit none
    character*150 :: fname_force_dry,fname
    integer :: iunit

    ! If variable_eta_init then function set_eta_init is called
    ! to set initial eta when interpolating onto newly refined patches

    iunit = 7
    fname = 'setprob.data'
!   # open the unit with new routine from Clawpack 4.4 to skip over
!   # comment lines starting with #:
    call opendatafile(iunit, fname)

    read(iunit,*) variable_eta_init
    read(iunit,*) use_force_dry

    if (use_force_dry) then
        read(iunit,*) tend_force_dry
        read(iunit,*) fname_force_dry
        !fname_force_dry = 'topo_force_dry.data'
        call read_force_dry(trim(fname_force_dry))
        endif

end subroutine setprob
