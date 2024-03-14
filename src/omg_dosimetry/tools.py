### This file is adapted from https://github.com/terranjp/matplotlib-tools/blob/master/matplotlib_tools/tools.py

from __future__ import print_function
import numpy as np
from matplotlib.widgets import AxesWidget
from matplotlib.text import Text


class Ruler(AxesWidget):
    """
    A ruler to measure distances and angles on an axes instance. 
    
    For the ruler to remain responsive you must keep a reference to it.

    Parameters
    ----------
    ax  : the  :class:`matplotlib.axes.Axes` instance 
       
    active : bool, default is True
        Whether the ruler is active or not. 
    
    length_unit  : string, A length unit identifier to use in displayed text  
        i.e. ('ft', or 'm')
                        
    angle_unit  : string, The type of angle unit ('degrees' or 'radians')
        
    print_text  : bool, default is False
        Whether the length measure string is printed to the console

    useblit : bool, default is False
        If True, use the backend-dependent blitting features for faster
        canvas updates. 

    lineprops : dict, default is None
      Dictionary of :class:`matplotlib.lines.Line2D` properties
                   
    markerprops : dict, default is None
      Dictionary of :class:`matplotlib.markers.MarkerStyle` properties
      
    textprops: dict, default is None
        Dictionary of :class:`matplotlib.text.Text` properties. To reposition the 
        textbox you can overide the defaults which position the box in the top left
        corner of the axes. 
          
    Usage:
    ----------

    1. Hold left click drag and release to draw the ruler in the axes.
      - Hold shift while dragging to lock the ruler to the horizontal axis.
      - Hold control while drawing to lock the ruler to the vertical axis. 

    2. Right click one of the markers to move the ruler. 

    The keyboard can be used to activate and deactivate the ruler and toggle 
    visibility of the line and text:

    'm' : Toggles the ruler on and off. 

    'ctl+m' : Toggles the visibility of the ruler and text. 
    
    Example
    ----------
    
    >>> xCoord = np.arange(0, 5, 1)
    >>> yCoord = [0, 1, -3, 5, -3]
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    
    >>> markerprops = dict(marker='o', markersize=5, markeredgecolor='red')
    >>> lineprops = dict(color='red', linewidth=2)
    
    >>> ax.grid(True)
    >>> ax.plot(xCoord, yCoord)
    
    >>> ruler = Ruler(ax=ax,
                  useblit=True,
                  markerprops=markerprops,
                  lineprops=lineprops)
    
    >>> plt.show()
    
    """

    def __init__(self,
                 ax,
                 active=True,
                 length_unit=None,
                 angle_unit='degree',
                 print_text=False,
                 useblit=False,
                 lineprops=None,
                 textprops=None,
                 markerprops=None):
        """
        Add a ruler to *ax*. If ``ruler_active=True``, the ruler will be 
        activated when the plot is first created. If ``ruler_unit`` is set the 
        string will be appended to the length text annotations. 

        """
        AxesWidget.__init__(self, ax)

        self.connect_events()

        self.ax = ax
        self.fig = ax.figure

        self._print_text = print_text
        self._visible = True
        self.active = active
        self.length_unit = length_unit
        self.angle_unit = angle_unit
        self.useblit = useblit and self.canvas.supports_blit

        self._mouse1_pressed = False
        self._mouse3_pressed = False
        self._shift_pressed = False
        self._control_pressed = False
        self._y0 = None
        self._x1 = None
        self._y1 = None
        self._line_start_coords = None
        self._line_end_coords = None
        self._ruler_marker = None
        self._background = None
        self._ruler_moving = False
        self._end_a_lock = False
        self._end_b_lock = False
        self._end_c_lock = False
        self._old_marker_a_coords = None
        self._old_marker_c_coords = None
        self._old_mid_coords = None

        if lineprops is None:
            lineprops = {}

        bbox = dict(facecolor='white',
                    alpha=0.5,
                    boxstyle='round',
                    edgecolor='0.75')

        used_textprops = dict(xy=(0, 1),
                              xytext=(10, -10),
                              xycoords='axes fraction',
                              textcoords='offset points',
                              ha='left',
                              va='center',
                              bbox=bbox)

        x0 = np.nan
        y0 = np.nan

        self._ruler, = self.ax.plot([x0, x0], [y0, y0], **lineprops)

        used_markerprops = dict(marker='s',
                                markersize=3,
                                markerfacecolor='white',
                                markeredgecolor='black',
                                markeredgewidth=0.5,
                                picker=5,
                                visible=False)

        # If marker or text  props are given as an argument combine with the
        # default marker props. Don't really want to override the entire props
        # if a user only gives one value.

        if markerprops is not None:
            used_markerprops.update(markerprops)

        if textprops is not None:
            used_textprops.update(used_textprops)

        self._axes_text = self.ax.annotate(text='', **used_textprops)
        self.ax.add_artist(self._axes_text)

        self._marker_a, = self.ax.plot((x0, y0), **used_markerprops)
        self._marker_b, = self.ax.plot((x0, y0), **used_markerprops)
        self._marker_c, = self.ax.plot((x0, y0), **used_markerprops)

        self._artists = [self._axes_text,
                         self._ruler,
                         self._marker_a,
                         self._marker_b,
                         self._marker_c]

    def connect_events(self):
        """
        Connect all events to the various callbacks
        """
        self.connect_event('button_press_event', self._on_press)
        self.connect_event('button_release_event', self._on_release)
        self.connect_event('motion_notify_event', self._on_move)
        self.connect_event('key_press_event', self._on_key_press)
        self.connect_event('key_release_event', self._on_key_release)

    def ignore(self, event):
        """
        Ignore events if the cursor is out of the axes or the widget is locked
        """
        if not self.canvas.widgetlock.available(self):
            return True
        if event.inaxes != self.ax.axes:
            return True
        if not self.active:
            return True

    def _on_key_press(self, event):
        """
        Handle key press events. 
        
        If shift is pressed the ruler will be constrained to horizontal axis
        If control is pressed the ruler will be constrained to vertical axis
        If m is pressed the ruler will be toggled on and off
        If ctrl+m is pressed the visibility of the ruler will be toggled
        """

        if event.key == 'shift':
            self._shift_pressed = True

        if event.key == 'control':
            self._control_pressed = True

        if event.key == 'm':
            self.toggle_ruler()

        if event.key == 'ctrl+m':
            self.toggle_ruler_visibility()

    def _on_key_release(self, event):
        """
        Handle key release event, flip the flags to false.
        """
        if event.key == 'shift':
            self._shift_pressed = False

        if event.key == 'control':
            self._control_pressed = False

    def toggle_ruler(self):
        """
        Called when the 'm' key is pressed. If ruler is on turn it off, and 
        vise versa
        """

        self.active = not self.active

    def toggle_ruler_visibility(self):
        """
        Called when the 'ctl+m' key is pressed. If ruler is visible turn it off
        , and vise versa
        """
        if self._visible is True:
            for artist in self._artists:
                artist.set_visible(False)
            self.active = False
            self._visible = False

        elif self._visible is False:
            for artist in self._artists:
                artist.set_visible(True)
            self._visible = True

        self.canvas.draw_idle()

    def _on_press(self, event):
        """
        On mouse button press check which button has been pressed and handle 
        """
        if self.ignore(event):
            return
        if event.button == 1 and self._mouse3_pressed is False:
            self._handle_button1_press(event)
        elif event.button == 3:
            self._handle_button3_press(event)

    def _handle_button1_press(self, event):
        """
        On button 1 press start drawing the ruler line from the initial 
        press position
        """

        self._mouse1_pressed = True
        self._x0 = event.xdata
        self._y0 = event.ydata
        self._marker_a.set_data((event.xdata, event.ydata))
        self._marker_a.set_visible(True)

        if self.useblit:
            self._marker_a.set_data(self._x0, self._y0)
            for artist in self._artists:
                artist.set_animated(True)
            self.canvas.draw()
            self._background = self.canvas.copy_from_bbox(self.fig.bbox)

    def _handle_button3_press(self, event):
        """
        If button 3 is pressed (right click) check if cursor is at one of the 
        ruler markers and the move the ruler accordingly. 
        """
        contains_a, _ = self._marker_a.contains(event)
        contains_b, _ = self._marker_b.contains(event)
        contains_c, _ = self._marker_c.contains(event)

        if not (contains_a or contains_b or contains_c):
            return

        self._end_a_lock = contains_a
        self._end_b_lock = contains_b
        self._end_c_lock = contains_c

        line_coords = self._ruler.get_path().vertices
        self._x0 = line_coords[0][0]
        self._y0 = line_coords[0][1]
        self._x1 = line_coords[1][0]
        self._y1 = line_coords[1][1]

        self._old_marker_a_coords = self._marker_a.get_path().vertices
        self._old_marker_c_coords = self._marker_c.get_path().vertices
        self._old_mid_coords = self.midline_coords

    def _on_move(self, event):
        """
        On motion draw the ruler if button 1 is pressed. If one of the markers
        is locked indicating move the ruler according to the locked marker
        """

        if event.inaxes != self.ax.axes:
            return

        if self._end_a_lock or self._end_b_lock or self._end_c_lock is True:
            self._move_ruler(event)

        if self._mouse1_pressed is True:
            self._draw_ruler(event)

    def _move_ruler(self, event):
        """
        If one of the markers is locked move the ruler according the selected
        marker. 
        """

        # This flag is used to prevent the ruler from clipping when a marker is
        # first selected
        if self._ruler_moving is False:
            if self.useblit:
                for artist in self._artists:
                    artist.set_animated(True)
                self.canvas.draw()
                self._background = self.canvas.copy_from_bbox(self.fig.bbox)
                self._ruler_moving = True

        if self._end_a_lock is True:
            # If marker a is locked only move end a.
            pos_a = event.xdata, self._x1
            pos_b = event.ydata, self._y1
            self._marker_a.set_data(event.xdata, event.ydata)
            self._ruler.set_data(pos_a, pos_b)
            self._set_midline_marker()

        if self._end_c_lock is True:
            # If marker a is locked only move end c.
            pos_a = self._x0, event.xdata
            pos_b = self._y0, event.ydata
            self._marker_c.set_data(event.xdata, event.ydata)
            self._ruler.set_data(pos_a, pos_b)
            self._set_midline_marker()

        if self._end_b_lock is True:
            # If marker b is locked shift the whole ruler.
            b_dx = event.xdata - self._old_mid_coords[0]
            b_dy = event.ydata - self._old_mid_coords[1]
            pos_a = self._x0 + b_dx, self._x1 + b_dx
            pos_b = self._y0 + b_dy, self._y1 + b_dy

            marker_a_coords = self._old_marker_a_coords[0][0] + b_dx, \
                              self._old_marker_a_coords[0][1] + b_dy
            marker_c_coords = self._old_marker_c_coords[0][0] + b_dx, \
                              self._old_marker_c_coords[0][1] + b_dy

            self._ruler.set_data(pos_a, pos_b)
            self._marker_a.set_data(marker_a_coords)
            self._marker_b.set_data(event.xdata, event.ydata)
            self._marker_c.set_data(marker_c_coords)

        self._update_text()
        self._update_artists()

    def _set_midline_marker(self):
        self._marker_b.set_visible(True)
        self._marker_b.set_data(self.midline_coords)

    @property
    def midline_coords(self):
        pos0, pos1 = self._ruler.get_path().vertices
        mid_line_coords = (pos0[0] + pos1[0]) / 2, (pos0[1] + pos1[1]) / 2
        return mid_line_coords

    def _draw_ruler(self, event):
        """
        If the left mouse button is pressed and held draw the ruler as the 
        mouse is dragged
        """

        self._x1 = event.xdata
        self._y1 = event.ydata

        # If shift is pressed ruler is constrained to horizontal axis
        if self._shift_pressed is True:
            pos_a = self._x0, self._x1
            pos_b = self._y0, self._y0
        # If control is pressed ruler is constrained to vertical axis
        elif self._control_pressed is True:
            pos_a = self._x0, self._x0
            pos_b = self._y0, self._y1
        # Else the ruler follow the mouse cursor
        else:
            pos_a = self._x0, self._x1
            pos_b = self._y0, self._y1

        self._ruler.set_data([pos_a], [pos_b])
        x1 = self._ruler.get_path().vertices[1][0]
        y1 = self._ruler.get_path().vertices[1][1]
        self._marker_c.set_visible(True)
        self._marker_c.set_data(x1, y1)
        self._set_midline_marker()
        self._update_text()
        self._update_artists()

    def _update_artists(self):
        if self.useblit:
            if self._background is not None:
                self.canvas.restore_region(self._background)
            else:
                self._background = self.canvas.copy_from_bbox(self.fig.bbox)

            for artist in self._artists:
                self.ax.draw_artist(artist)

            self.canvas.blit(self.fig.bbox)
        else:
            self.canvas.draw_idle()

    def _update_text(self):
        if self.length_unit is not None:
            detail_string = 'L: {:0.3f} {}; dx: {:0.3f} {}; dy: {:0.3f} {}; ' \
                            'angle: {:0.3f} deg'.format(self.ruler_length,
                                                        self.length_unit,
                                                        self.ruler_dx,
                                                        self.length_unit,
                                                        self.ruler_dy,
                                                        self.length_unit,
                                                        self.ruler_angle)
        else:
            detail_string = 'L: {:0.3f}; dx: {:0.3f}; dy: {:0.3f}; ' \
                            'ang: {:0.3f} deg'.format(self.ruler_length,
                                                        self.ruler_dx,
                                                        self.ruler_dy,
                                                        self.ruler_angle)

        self._axes_text.set_text(detail_string)
        if self._print_text is True:
            print(detail_string)

    def _on_release(self, event):
        self._mouse1_pressed = False
        self._mouse3_pressed = False
        self._ruler_moving = False
        self._end_a_lock = False
        self._end_b_lock = False
        self._end_c_lock = False

    @property
    def ruler_length(self):
        pos0, pos1 = self._ruler.get_path().vertices
        return np.hypot((pos1[0] - pos0[0]), (pos0[1] - pos1[1]))

    @property
    def ruler_dx(self):
        pos0, pos1 = self._ruler.get_path().vertices
        return pos1[0] - pos0[0]

    @property
    def ruler_dy(self):
        pos0, pos1 = self._ruler.get_path().vertices
        return pos1[1] - pos0[1]

    @property
    def ruler_angle(self):
        pos0, pos1 = self._ruler.get_path().vertices
        angle = np.arctan2(pos1[0] - pos0[0], pos1[1] - pos0[1])

        if self.angle_unit == 'degree':
            return angle * 180 / np.pi
        else:
            return angle


class TextMover(object):
    """
    A simple little tool to move text annotation in an axes by clicking and dragging them. 

    """

    def __init__(self, ax, active=True):
        self.ax = ax
        self.active = active
        self.selectedText = None
        self.mousePressed = False
        self.background = None
        self.connect_events()

    def connect_events(self):
        self.ax.figure.canvas.mpl_connect("pick_event", self.on_pick_event)
        self.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)

    def on_pick_event(self, event):

        if self.active is False:
            return

        if event.mouseevent.button != 1:
            return

        if isinstance(event.artist, Text):
            self.mousePressed = True
            self.selectedText = event.artist

            canvas = self.selectedText.figure.canvas
            axes = self.selectedText.axes
            self.selectedText.set_animated(True)
            canvas.draw()
            self.background = canvas.copy_from_bbox(self.selectedText.axes.bbox)

            axes.draw_artist(self.selectedText)
            canvas.blit(axes.bbox)

    def on_motion(self, event):
        if event.inaxes != self.ax.axes:
            return

        if self.mousePressed is True and self.selectedText:
            self.x1 = event.xdata
            self.y1 = event.ydata

            coords = (self.x1, self.y1)

            self.selectedText.set_position(coords)
            canvas = self.selectedText.figure.canvas
            axes = self.selectedText.axes
            canvas.restore_region(self.background)
            axes.draw_artist(self.selectedText)
            canvas.blit(axes.bbox)

    def on_release(self, event):

        if self.selectedText is None:
            return

        self.mousePressed = False
        self.selectedText.set_animated(False)

        self.background = None
        self.selectedText.figure.canvas.draw()
        self.selectedText = None

