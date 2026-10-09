import math
import pytest
import torch
from task1_dpo.dpo import dpo_loss
from task2_ppo.ppo import ppo_policy_loss,compute_gae,shaped_rewards
from task3_grpo.grpo import group_relative_advantages,grpo_policy_loss,mask_truncated_sequences


def test_dpo_equal_policy_and_reference_is_log_two_and_correct_gradient():
    c=torch.tensor([-2.,-3.],requires_grad=True);r=torch.tensor([-4.,-8.],requires_grad=True)
    loss,diag=dpo_loss(c,r,c.detach(),r.detach(),.3)
    assert loss.item()==pytest.approx(math.log(2))
    assert diag['preference_accuracy']==0
    loss.backward()
    assert torch.allclose(c.grad,torch.full_like(c,-.075))
    assert torch.allclose(r.grad,torch.full_like(r,.075))


def test_dpo_reference_margin_subtracted():
    c=torch.tensor([-1.]);r=torch.tensor([-3.]);rc=torch.tensor([-1.]);rr=torch.tensor([-5.])
    loss,_=dpo_loss(c,r,rc,rr,.1)
    assert loss==pytest.approx(math.log1p(math.exp(.2)))


def test_ppo_sign_sensitive_pessimistic_clipping_and_mask():
    ratios=torch.tensor([[1.5,.5,1.5,.5,3.]])
    advantage=torch.tensor([[1.,-1.,-1.,1.,100.]])
    mask=torch.tensor([[1.,1.,1.,1.,0.]])
    new=ratios.log().requires_grad_()
    loss,_,fraction=ppo_policy_loss(new,torch.zeros_like(new),advantage,mask,.2)
    # min: [1.2,-.8,-1.5,.5]; only two clipped terms have zero policy gradient.
    assert loss.item()==pytest.approx(.15)
    assert fraction.item()==1.
    loss.backward()
    assert new.grad[0,0]==0 and new.grad[0,1]==0 and new.grad[0,4]==0
    assert new.grad[0,2]>0 and new.grad[0,3]<0


def test_gae_zero_bootstrap_at_last_valid_token():
    rewards=torch.tensor([[0.,1.,99.],[2.,99.,99.]])
    values=torch.tensor([[.2,.3,99.],[.5,99.,99.]])
    mask=torch.tensor([[1.,1.,0.],[1.,0.,0.]])
    adv,ret=compute_gae(rewards,values,mask,1.,.95)
    assert torch.allclose(adv,torch.tensor([[.765,.7,0.],[1.5,0.,0.]]))
    assert ret[0,1]==1 and ret[1,0]==2


def test_kl_shaping_and_terminal_reward_respect_response_mask():
    lp=torch.tensor([[.4,.6,5.]])
    mask=torch.tensor([[1.,1.,0.]])
    rewards=shaped_rewards(torch.tensor([2.]),lp,torch.zeros_like(lp),mask,.1)
    assert torch.allclose(rewards,torch.tensor([[-.04,1.94,0.]]))


def test_grpo_within_prompt_invariance_and_zero_variance():
    rewards=torch.tensor([1.,3.,100.,100.,5.]);gids=torch.tensor([4,4,9,9,22])
    advantages=group_relative_advantages(rewards,gids)
    assert torch.allclose(advantages,torch.tensor([-1.,1.,0.,0.,0.]),atol=2e-6)
    offsets=torch.tensor([1000.,1000.,-90.,-90.,0.])
    assert torch.allclose(advantages,group_relative_advantages(rewards+offsets,gids))
    permutation=torch.tensor([4,2,0,3,1])
    assert torch.allclose(advantages[permutation],group_relative_advantages(rewards[permutation],gids[permutation]))


def test_grpo_normalization_direct_gradient_allocation_and_truncation():
    mask=torch.tensor([[1.,0.,0.,0.],[1.,1.,1.,1.]])
    new=torch.zeros_like(mask,requires_grad=True);old=torch.zeros_like(mask)
    args=(new,old,torch.ones(2),mask,old,.2,0.)
    canonical,_=grpo_policy_loss(*args,loss_type='grpo')
    gc=torch.autograd.grad(canonical,new,retain_graph=True)[0]
    dr,_=grpo_policy_loss(*args,loss_type='dr_grpo',max_completion_length=4)
    gd=torch.autograd.grad(dr,new)[0]
    assert gc[0,0].abs()==pytest.approx(.5)
    assert gc[1,0].abs()==pytest.approx(.125)
    assert gd[0,0].abs()==gd[1,0].abs()==pytest.approx(.125)
    assert not mask_truncated_sequences(mask,[True,False])[0].any()
