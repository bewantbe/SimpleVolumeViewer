# TVTK demo
# from # https://wizardforcel.gitbooks.io/hyry-studio-scipy/content/8.html

from tvtk.api import tvtk

cone = tvtk.ConeSource( height=3.0, radius=1.0, resolution=10 )
# Error: traits.trait_errors.TraitError: The 'input' trait of a PolyDataMapper instance is 'read only'.
cone_mapper = tvtk.PolyDataMapper( input = cone.output )
cone_actor = tvtk.Actor( mapper=cone_mapper )
cone_actor.property.representation = "w"

ren1 = tvtk.Renderer()
ren1.add_actor( cone_actor )
ren1.background = 0.1, 0.2, 0.4
ren_win = tvtk.RenderWindow()
ren_win.add_renderer( ren1 )
ren_win.size = 300, 300

iren = tvtk.RenderWindowInteractor( render_window = ren_win )
iren.initialize()
iren.start()
