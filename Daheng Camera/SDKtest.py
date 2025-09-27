import gxipy as gx

dm = gx.DeviceManager()
n, _ = dm.update_all_device_list()
print("dev_num =", n)
cam = dm.open_device_by_index(1)

fc = cam.get_remote_device_feature_control()
cam.stream_off()
fc.get_int_feature("OffsetX").set(0)
fc.get_int_feature("OffsetY").set(0)
fc.get_int_feature("Width").set(304)
fc.get_int_feature("Height").set(304)

cam.stream_on()
raw = cam.data_stream[0].get_image()
print("frame:", raw.get_frame_id(), raw.get_width(), raw.get_height())
cam.stream_off()
cam.close_device()
