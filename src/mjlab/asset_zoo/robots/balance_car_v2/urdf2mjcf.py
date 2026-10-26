import mujoco

model = mujoco.MjModel.from_xml_path('/home/yun/data/git_projects/mjlab/src/mjlab/asset_zoo/robots/balance_car_v2/xmls/balance_car.urdf')
mujoco.mj_saveLastXML('/home/yun/data/git_projects/mjlab/src/mjlab/asset_zoo/robots/balance_car_v2/xmls/balance_car2.xml', model)