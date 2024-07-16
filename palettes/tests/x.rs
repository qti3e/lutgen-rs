use std::mem::transmute;

#[test]
fn x() {
    for i in 0..554u16 {
        let x: lutgen_palettes::Palette = unsafe { transmute(i) };
        if x.get().len() == 16 {
            println!("{x:?}");
            return;
        }
    }
}
