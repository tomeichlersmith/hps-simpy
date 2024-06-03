"""an awkward "schema" for my HPS analysis tuples

Each event consists of a single reconstructed Vertex,
the electron and positron it is made out of,
and some other categorical/event-wide variables.
If there is an 'mc' branch in the input tree,
they are read in as well and formed into a branch of MCParticles.
"""

import uproot
import awkward as ak

import vector
vector.register_awkward()


def _three_vector(branch_fn, pre_coord, post_coord=''):
    return ak.zip({
        c: branch_fn(f'{pre_coord}{c}{post_coord}')
        for c in ['x', 'y', 'z']
    }, with_name='Vector3D')


def TVector3D_to_Vector3D(tv):
    return ak.zip({
        c: tv[f'f{c.upper()}']
        for c in ['x','y','z']
    }, with_name='Vector3D')


@ak.mixin_class(ak.behavior)
class MCParticle:
    @classmethod
    def from_tree(cls, tree, basename, **array_kwargs):
        def _branch(name, postfix=''):
            return tree[f'{basename}.{name}{postfix}'].array(**array_kwargs)
        form = {
            m: _branch(m, postfix='_')
            for m in [
                'id', 'n_daughters', 'charge', 'pdg', 'momPDG',
                'gen', 'sim', 'energy', 'mass', 'time'
            ]
        }
        form.update({
            'p' : _three_vector(_branch, 'p', '_'),
            'vtx' : _three_vector(_branch, 'vtx_', '_'),
            'p_ep' : _three_vector(_branch, 'p', '_ep'),
            'ep' : _three_vector(_branch, 'ep_', '_')
        })
        return ak.zip(form, with_name=cls.__name__, depth_limit=2)


@ak.mixin_class(ak.behavior)
class Cluster:
    @classmethod
    def from_tree(cls, tree, basename, **array_kwargs):
        def _branch(name):
            return tree[f'{basename}.{name}_'].array(**array_kwargs)
        return ak.zip({
            m: _branch(m)
            for m in ['seed_hit', 'x', 'y', 'z', 'energy', 'time']
        }, with_name=cls.__name__)


@ak.mixin_class(ak.behavior)
class Track:
    @classmethod
    def from_tree(cls, tree, basename, **array_kwargs):
        def _branch(name):
            return tree[f'{basename}.{name}_'].array(**array_kwargs)
        form = {
            m: _branch(m)
            for m in [
                'n_hits', 'track_volume', 'type', 'd0', 'phi0',
                'omega', 'tan_lambda', 'z0', 'chi2', 'ndf',
                'id', 'charge', 'nShared', 'SharedLy0', 'SharedLy1'
            ]
        }
        # trk_dict.update({
        #    m : branch(f'{name}_.track_.{m}_[14]')
        #    for m in ['isolation','lambda_kinks','phi_kinks']
        # })
        form.update({
            'time': _branch('track_time'),
            'p': _three_vector(_branch, 'p', ''),
            'pos_at_ecal': _three_vector(_branch, '', '_at_ecal')
        })
        return ak.zip(form, with_name=cls.__name__)


@ak.mixin_class(ak.behavior)
class Particle:
    
    @classmethod
    def from_tree(cls, tree, basename, **array_kwargs):
        def _branch(name):
            return tree[f'{basename}/{name}_'].array(**array_kwargs)
        form = {
            m: _branch(m)
            for m in [
                'charge','type','pdg','goodness_pid',
                'energy','mass'
            ]
        }
        form.update({
            'track' : Track.from_tree(tree, f'{basename}/track_', **array_kwargs),
            'cluster' : Cluster.from_tree(tree, f'{basename}/cluster_', **array_kwargs),
            'p': _three_vector(_branch, 'p', ''),
            'p_corr': _three_vector(_branch, 'p', '_corr')
        })
        return ak.zip(form, depth_limit=1, with_name=cls.__name__);


@ak.mixin_class(ak.behavior)
class Vertex:
    @classmethod
    def from_tree(cls, tree, basename, **array_kwargs):
        def _branch(name):
            return tree[f'{basename}/{name}_'].array(**array_kwargs)
        form = {
            m: _branch(m)
            for m in [
                'chi2', 'ndf', 'invM', 'invMerr', 'probability', 'id',
                'type', 'parameters'
            ]
        }
        form.update({
            v : TVector3D_to_Vector3D(_branch(v))
            for v in [
                'pos', 'p1', 'p2', 'p'
            ]
        })
        form.update({
            'invM_smear' : tree[f'{basename}_invm_smear'].array(**array_kwargs),
        })
        return ak.zip(form, depth_limit=1, with_name=cls.__name__)


@ak.mixin_class(ak.behavior)
class Event:
    @classmethod
    def from_tree(cls, tree, **array_kwargs):
        def _branch(name):
            return tree[name].array(**array_kwargs)
        form = {
            m: _branch(m)
            for m in [
                'weight', 'eleL1', 'eleL2',
                'posL1', 'posL2', 'vtx_proj_sig',
                'vtx_proj_x', 'vtx_proj_x_sig',
                'vtx_proj_y', 'vtx_proj_y_sig'
            ]
        }
        form.update({
            'vertex': Vertex.from_tree(tree, 'vertex', **array_kwargs),
            'ele': Particle.from_tree(tree, 'ele', **array_kwargs),
            'pos': Particle.from_tree(tree, 'pos', **array_kwargs),
        })
        if 'mc' in tree.keys(recursive=False):
            form.update({
                'mc' : MCParticle.from_tree(tree, 'mc', **array_kwargs)
            })
        return ak.zip(form, depth_limit=1, with_name=cls.__name__)


def load_events(fp, tree_name = 'preselection', **array_kwargs):
    with uproot.open(fp) as f:
        return Event.from_tree(f[tree_name], **array_kwargs)