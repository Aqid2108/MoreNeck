def test_library_low_level_classes_import():
    from st3215.group_sync_read import GroupSyncRead
    from st3215.group_sync_write import GroupSyncWrite
    from st3215.port_handler import PortHandler
    from st3215.protocol_packet_handler import protocol_packet_handler
    from st3215.values import COMM_SUCCESS, STS_PRESENT_POSITION_L

    assert STS_PRESENT_POSITION_L == 0x38
    assert COMM_SUCCESS == 0
    assert all([GroupSyncRead, GroupSyncWrite, PortHandler, protocol_packet_handler])


def test_package_imports():
    import neck

    assert neck.__doc__
