import mujoco

model = mujoco.MjModel.from_xml_path('D:\\Git-Project\\mjlab\\src\\mjlab\\asset_zoo\\robots\\balance_car_v2\\xmls\\balance_car.urdf')
mujoco.mj_saveLastXML('D:\\Git-Project\\mjlab\\src\\mjlab\\asset_zoo\\robots\\balance_car_v2\\xmls\\balance_car.xml', model)