subroutine setprob()

    use topo_module, only: variable_eta_init
    use qinit_module, only: read_wet_mask, use_wet_mask, t_stays_dry
    implicit none
    character*150 :: fname_wet_mask,fname
    integer :: iunit

    ! If variable_eta_init then function set_eta_init is called
    ! to set initial eta when interpolating onto newly refined patches

    iunit = 7
    fname = 'setprob.data'
!   # open the unit with new routine from Clawpack 4.4 to skip over
!   # comment lines starting with #:
    call opendatafile(iunit, fname)

    read(iunit,*) variable_eta_init
    read(iunit,*) use_wet_mask

    if (use_wet_mask) then
        read(iunit,*) t_stays_dry
        read(iunit,*) fname_wet_mask
        !fname_wet_mask = 'topo_wet_mask.data'
        call read_wet_mask(trim(fname_wet_mask))
        endif

end subroutine setprob
