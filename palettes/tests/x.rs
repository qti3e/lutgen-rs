use std::mem::transmute;

#[test]
fn x() {
    let mut count = 0;

    for i in 0..554u16 {
        let x: lutgen_palettes::Palette = unsafe { transmute(i) };
        if x.get().len() % 8 == 0 {
            count += 1;
        }
    }

    println!("{count}");
}
