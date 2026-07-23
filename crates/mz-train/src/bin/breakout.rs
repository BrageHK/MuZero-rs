use std::num::NonZeroU32;
use std::time::{Duration, Instant};

use ale::{Ale, BundledRom};

use winit::dpi::{LogicalPosition, LogicalSize, PhysicalSize};
use winit::event::{Event, VirtualKeyCode};
use winit::event_loop::{ControlFlow, EventLoop};
use winit_input_helper::WinitInputHelper;

const SCREEN_WIDTH: u32 = 160;
const SCREEN_HEIGHT: u32 = 210;

const FRAME_DURATION: Duration = Duration::from_nanos(1_000_000_000 / 60);

fn main() {
	let mut ale = Ale::new();
	ale.load_rom(BundledRom::Breakout).expect("Illegal rom");

	let event_loop = EventLoop::new();
	let mut input = WinitInputHelper::new();
	let (window, mut hidpi_factor) =
		create_window("Breakout", &event_loop, SCREEN_WIDTH, SCREEN_HEIGHT);

	let context = unsafe { softbuffer::Context::new(&window) }.expect("create context");
	let mut surface = unsafe { softbuffer::Surface::new(&context, &window) }.expect("create surface");

	let mut paused = false;
	let mut prev_update = Instant::now();
	println!("=== CONTROLS ===");
	println!("Space - Start");
	println!("A / Left - Flipper left");
	println!("D / Right - Flipper right");
	println!("P - Toggle pause");
	println!();
	println!("Paused: false");

	let mut screen = vec![0u8; SCREEN_WIDTH as usize * SCREEN_HEIGHT as usize * 3];

	event_loop.run(move |event, _, control_flow| {
		if let Event::RedrawRequested(_) = event {
			let screen_width = ale.screen_width();
			let screen_height = ale.screen_height();
			screen.resize(screen_width * screen_height * 3, 0);
			ale.get_screen_rgb(&mut screen);

			let win_size = window.inner_size();
			let (Some(win_w), Some(win_h)) =
				(NonZeroU32::new(win_size.width), NonZeroU32::new(win_size.height))
			else {
				return;
			};
			surface.resize(win_w, win_h).expect("resize surface");

			let win_w = win_w.get() as usize;
			let win_h = win_h.get() as usize;
			let mut buffer = surface.buffer_mut().expect("get buffer");
			for dy in 0..win_h {
				let sy = dy * screen_height / win_h;
				for dx in 0..win_w {
					let sx = dx * screen_width / win_w;
					let si = (sy * screen_width + sx) * 3;
					let r = screen[si] as u32;
					let g = screen[si + 1] as u32;
					let b = screen[si + 2] as u32;
					buffer[dy * win_w + dx] = (r << 16) | (g << 8) | b;
				}
			}
			buffer.present().expect("present buffer");
		}

		if input.update(&event) {
			if input.key_pressed(VirtualKeyCode::Escape) || input.close_requested() || input.destroyed() {
				*control_flow = ControlFlow::Exit;
				return;
			}
			if input.key_pressed(VirtualKeyCode::P) {
				paused = !paused;
				println!("Paused: {}", paused);
				if !paused {
					prev_update = Instant::now();
				}
			}
			if input.key_pressed(VirtualKeyCode::R) {
				ale.reset_game();
				println!("RESET");
			}

			let action = if input.key_held(VirtualKeyCode::Left) || input.key_held(VirtualKeyCode::A) {
				Some(4)
			} else if input.key_held(VirtualKeyCode::Right) || input.key_held(VirtualKeyCode::D) {
				Some(3)
			} else if input.key_held(VirtualKeyCode::Space) {
				Some(1)
			} else if !paused {
				Some(0)
			} else {
				None
			};

			let now = Instant::now();
			let mut diff = now - prev_update;
			if diff > 5 * FRAME_DURATION {
				diff = 5 * FRAME_DURATION;
				println!("Warning: skip of {}s occured", (diff - 5 * FRAME_DURATION).as_secs_f64());
			}
			while diff > FRAME_DURATION {
				diff -= FRAME_DURATION;
				prev_update = now;
				if let Some(action) = action {
					if ale.legal_action_set().contains(&action) {
						ale.act(action);
					} else {
						println!("Warning: illegal action: {}", action);
					}
				}
			}

			if ale.is_game_over() {
				println!("Game OVER");
				ale.reset_game();
				println!("RESET");
			}

			if let Some(factor) = input.scale_factor_changed() {
				hidpi_factor = factor;
			}
			let _ = hidpi_factor;
			if input.window_resized().is_some() {
				window.request_redraw();
			}

			window.request_redraw();
		}
	});
}

fn create_window(
	title: &str,
	event_loop: &EventLoop<()>,
	width: u32,
	height: u32,
) -> (winit::window::Window, f64) {
	let window = winit::window::WindowBuilder::new()
		.with_visible(false)
		.with_title(title)
		.build(event_loop)
		.unwrap();
	let hidpi_factor = window.scale_factor();

	let width = width as f64;
	let height = height as f64;
	let (monitor_width, monitor_height) = {
		let size = window.current_monitor().map(|m| m.size()).unwrap_or(PhysicalSize::new(1920, 1080));
		(size.width as f64 / hidpi_factor, size.height as f64 / hidpi_factor)
	};
	let scale = (monitor_height / height * 2.0 / 3.0).round().max(1.0);

	let min_size: LogicalSize<f64> = PhysicalSize::new(width, height).to_logical(hidpi_factor);
	let default_size = LogicalSize::new(width * scale, height * scale);
	let center = LogicalPosition::new(
		(monitor_width - width * scale) / 2.0,
		(monitor_height - height * scale) / 2.0,
	);
	window.set_inner_size(default_size);
	window.set_min_inner_size(Some(min_size));
	window.set_outer_position(center);
	window.set_visible(true);

	(window, hidpi_factor)
}
