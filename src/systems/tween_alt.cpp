#include "welvet/systems/tween_alt.hpp"

#include "welvet/systems/tween.hpp"
#include "welvet/systems/tween_split.hpp"

#include <string>

namespace welvet::systems::tween_alt {

Status apply_alt_child(stack::Child& c, const TensorF32& grad_out, const TensorF32& input,
                       double lr, int alt_times) {
  const int times = alt_times < 1 ? 1 : alt_times;
  const double soft_lr = lr * 0.5;
  for (int i = 0; i < times; ++i) {
    if (auto st = tween_split::apply_split_child(c, grad_out, input, lr,true); !st) {
      return Status::Err(st.code, std::string("tween_alt split: ") + st.message);
    }
    if (auto st = tween::apply_collected(c, grad_out, input, soft_lr); !st) {
      return Status::Err(st.code, std::string("tween_alt tween: ") + st.message);
    }
  }
  return Status::Ok();
}

Status apply_cell_alt(architecture::Cell& cell, const TensorF32& grad_out, const TensorF32& input,
                      const TensorF32& pre, const TensorF32& post, double lr, int alt_times) {
  const int times = alt_times < 1 ? 1 : alt_times;
  for (int i = 0; i < times; ++i) {
    if (auto st = tween_split::apply_cell_split(cell, grad_out, input, pre, post, lr); !st) {
      return Status::Err(st.code, std::string("tween_alt cell split: ") + st.message);
    }
    if (auto st = tween::apply_cell_projected(cell, grad_out, input, pre, post, lr * 0.5); !st) {
      return Status::Err(st.code, std::string("tween_alt cell tween: ") + st.message);
    }
  }
  return Status::Ok();
}

Status train_volume_mse(architecture::Grid& g, const TensorF32& input, const TensorF32& target,
                        double lr, double* loss, int alt_times, runtime::forward::Result* last_fwd) {
  if (!loss) {
    return Status::Err(StatusCode::InvalidArgument, "tween_alt: nil loss");
  }
  const int times = alt_times < 1 ? 1 : alt_times;
  double last = 0.0;
  for (int i = 0; i < times; ++i) {
    runtime::forward::Result fwd;
    if (auto st = runtime::forward::volume_dense(g, input, &fwd); !st) {
      return Status::Err(st.code, std::string("tween_alt fwd split: ") + st.message);
    }
    if (auto st = tween_split::train_volume_mse(g, fwd, target, lr, &last); !st) {
      return Status::Err(st.code, std::string("tween_alt split: ") + st.message);
    }
    if (auto st = runtime::forward::volume_dense(g, input, &fwd); !st) {
      return Status::Err(st.code, std::string("tween_alt fwd tween: ") + st.message);
    }
    if (auto st = tween::train_volume_mse(g, fwd, target, lr, &last); !st) {
      return Status::Err(st.code, std::string("tween_alt tween: ") + st.message);
    }
    if(last_fwd)*last_fwd=fwd;
  }
  *loss = last;
  return Status::Ok();
}

} // namespace welvet::systems::tween_alt
